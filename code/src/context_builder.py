"""Stage 2: relational context enrichment for the message router.

Loads the background dataset CSVs once and answers per-message lookups:
user notification behavior, group membership standing, business
relationship/verification, and historical evidence between a user and
whoever sent the message.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DATASET_DIR as DEFAULT_DATASET_DIR
from .config import (
    EVIDENCE_MIN_SIMILARITY,
    EVIDENCE_RELATIONSHIP_BOOST,
    EVIDENCE_TOP_N,
    NOTIFICATION_LOAD_DAYS,
)


def is_within_quiet_hours(window: str, when: datetime) -> bool:
    """Return True if `when` falls inside a "HH:MM-HH:MM" do-not-disturb window."""
    try:
        start_str, end_str = window.split("-")
        start = datetime.strptime(start_str.strip(), "%H:%M").time()
        end = datetime.strptime(end_str.strip(), "%H:%M").time()
    except (ValueError, AttributeError):
        return False

    now = when.time()
    if start <= end:
        return start <= now < end
    # Overnight windows (e.g. 22:00-07:00) wrap past midnight.
    return now >= start or now < end


class ContextBuilder:
    """Loads dataset CSVs with pandas and answers per-message context lookups."""

    def __init__(self, dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.users = self._read_csv("users.csv").set_index("user_id")
        self.groups = self._read_csv("groups.csv").set_index("group_id")
        self.group_members = self._read_csv("group_members.csv")
        self.business_accounts = self._read_csv("business_accounts.csv").set_index("business_id")
        self.user_business_history = self._read_csv("user_business_history.csv")
        self.message_history = self._read_csv("message_history.csv")
        self.message_events = self._read_csv("message_events.csv")
        self.daily_notification_summary = self._read_csv("daily_notification_summary.csv")

        self._history_by_user = dict(tuple(self.message_history.groupby("user_id")))
        # Per-user fitted TF-IDF (vectorizer, history matrix); None marks a
        # history whose vocabulary was empty.
        self._tfidf_by_user: dict[str, tuple[TfidfVectorizer, Any] | None] = {}

    def _read_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.dataset_dir / filename)

    # ---- 1. user quiet hours + activity habits ----------------------------

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        if user_id not in self.users.index:
            return {"user_id": user_id, "known": False}

        row = self.users.loc[user_id]
        opened = int(row["messages_opened_30d"])
        replied = int(row["messages_replied_30d"])
        dismissed = int(row["notifications_dismissed_30d"])
        reported = int(row["messages_reported_30d"])
        total_notifications = opened + dismissed

        return {
            "user_id": user_id,
            "known": True,
            "do_not_disturb_window": row["do_not_disturb_window"],
            "messages_opened_30d": opened,
            "messages_replied_30d": replied,
            "notifications_dismissed_30d": dismissed,
            "messages_reported_30d": reported,
            "open_rate_30d": round(opened / total_notifications, 3) if total_notifications else None,
            "reply_rate_30d": round(replied / opened, 3) if opened else None,
            "report_rate_30d": round(reported / total_notifications, 3) if total_notifications else None,
            "recent_notification_load": self.get_recent_notification_load(user_id),
        }

    def get_recent_notification_load(
        self, user_id: str, days: int = NOTIFICATION_LOAD_DAYS
    ) -> dict[str, Any]:
        rows = self.daily_notification_summary[self.daily_notification_summary["user_id"] == user_id]
        if rows.empty:
            return {"days_observed": 0, "avg_sent": 0.0, "avg_dismissed": 0.0, "dismiss_rate": None}

        rows = rows.sort_values("date", ascending=False).head(days)
        total_sent = int(rows["notifications_sent"].sum())
        total_dismissed = int(rows["notifications_dismissed"].sum())
        return {
            "days_observed": len(rows),
            "avg_sent": round(rows["notifications_sent"].mean(), 2),
            "avg_dismissed": round(rows["notifications_dismissed"].mean(), 2),
            "dismiss_rate": round(total_dismissed / total_sent, 3) if total_sent else None,
        }

    # ---- 2. group status ----------------------------------------------------

    def get_group_context(self, group_id: str, user_id: str) -> dict[str, Any]:
        if group_id not in self.groups.index:
            return {"group_id": group_id, "known": False}

        group_row = self.groups.loc[group_id]
        membership_rows = self.group_members[
            (self.group_members["group_id"] == group_id) & (self.group_members["user_id"] == user_id)
        ]

        context: dict[str, Any] = {
            "group_id": group_id,
            "known": True,
            "group_name": group_row["group_name"],
            "group_type": group_row["group_type"],
            "member_count": int(group_row["member_count"]),
            "admin_count": int(group_row["admin_count"]),
            "group_messages_30d": int(group_row["messages_30d"]),
            "user_is_member": not membership_rows.empty,
            "user_role": None,
            "user_muted_group": False,
            "user_messages_sent_30d": None,
            "user_messages_read_30d": None,
            "user_replies_sent_30d": None,
            "user_notifications_dismissed_30d": None,
        }

        if not membership_rows.empty:
            member = membership_rows.iloc[0]
            context.update(
                {
                    "user_role": member["role"],
                    "user_muted_group": bool(member["group_muted_by_user"]),
                    "user_messages_sent_30d": int(member["messages_sent_30d"]),
                    "user_messages_read_30d": int(member["messages_read_30d"]),
                    "user_replies_sent_30d": int(member["replies_sent_30d"]),
                    "user_notifications_dismissed_30d": int(member["notifications_dismissed_30d"]),
                }
            )

        return context

    # ---- 3. business account status -----------------------------------------

    def get_business_context(self, business_id: str, user_id: str) -> dict[str, Any]:
        if business_id not in self.business_accounts.index:
            return {"business_id": business_id, "known": False}

        biz = self.business_accounts.loc[business_id]
        context: dict[str, Any] = {
            "business_id": business_id,
            "known": True,
            "display_name": biz["display_name"],
            "brand_name": biz["brand_name"],
            "category": biz["category"],
            "verified": bool(biz["verified"]),
            "official_domain": biz["official_domain"],
            "domain_used_by_sender": biz["domain_used_by_sender"],
            "sender_domain_matches_official": biz["official_domain"] == biz["domain_used_by_sender"],
            "account_age_days": int(biz["account_age_days"]),
            "domain_used_by_sender_age_days": int(biz["domain_used_by_sender_age_days"]),
            "business_messages_sent_30d": int(biz["messages_sent_30d"]),
            "business_user_reports_30d": int(biz["user_reports_30d"]),
            "user_has_relationship": False,
        }

        history_rows = self.user_business_history[
            (self.user_business_history["user_id"] == user_id)
            & (self.user_business_history["business_id"] == business_id)
        ]
        if not history_rows.empty:
            # A user can carry more than one history row per business; keep the most recent.
            relationship = history_rows.sort_values("last_activity_at", ascending=False).iloc[0]
            context.update(
                {
                    "user_has_relationship": True,
                    "why_user_knows_account": relationship["why_user_knows_account"],
                    "last_activity_at": relationship["last_activity_at"],
                    "allows_promotions": bool(relationship["allows_promotions"]),
                    "promotions_opted_out_at": _none_if_nan(relationship["promotions_opted_out_at"]),
                    "activity_count_180d": int(relationship["activity_count_180d"]),
                    "messages_opened_30d": int(relationship["messages_opened_30d"]),
                    "messages_dismissed_30d": int(relationship["messages_dismissed_30d"]),
                    "messages_replied_30d": int(relationship["messages_replied_30d"]),
                    "last_reply_at": _none_if_nan(relationship["last_reply_at"]),
                }
            )

        return context

    # ---- 4. historical evidence between user and sender ---------------------

    def _similarity_scores(self, user_id: str, query_text: str) -> pd.Series | None:
        """Cosine similarity of query_text against the user's whole history.

        Returns a Series indexed like the user's history rows, or None when
        TF-IDF is unusable (no query text / empty history vocabulary).
        """
        if not query_text or not query_text.strip():
            return None
        history = self._history_by_user.get(user_id)
        if history is None or history.empty:
            return None

        if user_id not in self._tfidf_by_user:
            texts = history["message_text"].fillna("").tolist()
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                matrix = vectorizer.fit_transform(texts)
                self._tfidf_by_user[user_id] = (vectorizer, matrix)
            except ValueError:  # empty vocabulary
                self._tfidf_by_user[user_id] = None

        fitted = self._tfidf_by_user[user_id]
        if fitted is None:
            return None
        vectorizer, matrix = fitted
        query_vector = vectorizer.transform([query_text])
        return pd.Series(cosine_similarity(query_vector, matrix).ravel(), index=history.index)

    def get_historical_evidence(
        self,
        user_id: str,
        sender_user_id: str | None = None,
        business_id: str | None = None,
        group_id: str | None = None,
        query_text: str | None = None,
        top_n: int = EVIDENCE_TOP_N,
    ) -> list[dict[str, Any]]:
        history = self._history_by_user.get(user_id)
        if history is None or history.empty:
            return []

        # Content-based retrieval first: TF-IDF cosine similarity across the
        # user's whole history (catches e.g. the same scam text arriving from a
        # different sender), boosted for the current sender relationship.
        subset = pd.DataFrame()
        scores = self._similarity_scores(user_id, query_text or "")
        if scores is not None:
            boosted = scores.copy()
            if sender_user_id:
                boosted[history["sender_user_id"] == sender_user_id] += EVIDENCE_RELATIONSHIP_BOOST
            if business_id:
                boosted[history["business_id"] == business_id] += EVIDENCE_RELATIONSHIP_BOOST
            if group_id:
                boosted[history["group_id"] == group_id] += EVIDENCE_RELATIONSHIP_BOOST
            relevant = boosted[boosted > EVIDENCE_MIN_SIMILARITY]
            if not relevant.empty:
                subset = history.loc[relevant.sort_values(ascending=False).head(top_n).index]

        # Fallback: most specific relationship available, most recent first.
        if subset.empty:
            if sender_user_id:
                subset = history[history["sender_user_id"] == sender_user_id]
            elif business_id:
                subset = history[history["business_id"] == business_id]
            elif group_id:
                subset = history[history["group_id"] == group_id]
            else:
                return []
            if subset.empty:
                return []
            subset = subset.sort_values("created_at", ascending=False).head(top_n)

        events = self.message_events[self.message_events["user_id"] == user_id].set_index("message_id")

        evidence = []
        for _, row in subset.iterrows():
            message_id = row["message_id"]
            event = events.loc[message_id] if message_id in events.index else None
            evidence.append(
                {
                    "message_id": message_id,
                    "created_at": row["created_at"],
                    "message_text": row["message_text"] if pd.notna(row["message_text"]) else "",
                    "media_type": _none_if_nan(row["media_type"]),
                    "was_opened": bool(event["message_opened"]) if event is not None else None,
                    "was_replied": bool(event["message_replied"]) if event is not None else None,
                    "was_dismissed": bool(event["notification_dismissed"]) if event is not None else None,
                    "muted_after": bool(event["muted_after_message"]) if event is not None else None,
                    "was_reported": bool(event["message_reported"]) if event is not None else None,
                }
            )
        return evidence

    # ---- entry point used by the rule engine / LLM router --------------------

    def build_context(self, message: pd.Series, media_text: str | None = None) -> dict[str, Any]:
        """Assemble the full enrichment context for one row of messages.csv.

        media_text (an image extraction or voice transcript) joins the message
        text as the TF-IDF query so media messages also get relevant evidence.
        """
        user_id = message["user_id"]
        conversation_type = message.get("conversation_type")
        group_id = message.get("group_id")
        business_id = message.get("business_id")
        sender_user_id = message.get("sender_user_id")

        context: dict[str, Any] = {"user": self.get_user_profile(user_id)}

        if conversation_type == "group" and pd.notna(group_id):
            context["group"] = self.get_group_context(group_id, user_id)

        if conversation_type == "business" and pd.notna(business_id):
            context["business"] = self.get_business_context(business_id, user_id)

        text_parts = [
            part
            for part in (message.get("message_text"), media_text)
            if isinstance(part, str) and part.strip()
        ]
        context["evidence"] = self.get_historical_evidence(
            user_id=user_id,
            sender_user_id=sender_user_id if pd.notna(sender_user_id) else None,
            business_id=business_id if pd.notna(business_id) else None,
            group_id=group_id if pd.notna(group_id) else None,
            query_text="\n".join(text_parts),
        )
        return context


def _none_if_nan(value: Any) -> Any:
    return None if pd.isna(value) else value
