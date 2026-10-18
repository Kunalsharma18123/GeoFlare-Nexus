from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import engine
from backend.models import FireEvent
from backend.db_enrichment import update_event_intelligence


def main():
    with Session(engine) as session:
        event_ids = session.scalars(
            select(FireEvent.id).order_by(FireEvent.id)
        ).all()

        total = len(event_ids)

        print(f"Found {total} events.")
        print("Starting intelligence backfill...")
        print()

        success = 0
        failed = 0

        for index, event_id in enumerate(event_ids, start=1):
            try:
                result = update_event_intelligence(session, event_id)

                success += 1

                print(
                    f"[{index}/{total}] "
                    f"Event {event_id}: "
                    f"{result['predicted_class']} | "
                    f"{result['prediction_confidence']:.3f} | "
                    f"{result['risk_level']} "
                    f"({result['risk_score']:.2f})"
                )

            except Exception as exc:
                failed += 1

                print(
                    f"[{index}/{total}] "
                    f"Event {event_id}: FAILED | {exc}"
                )

        print()
        print("Backfill complete.")
        print(f"Total:   {total}")
        print(f"Success: {success}")
        print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()