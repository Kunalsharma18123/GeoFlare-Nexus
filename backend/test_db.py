from sqlalchemy import inspect

from backend.db import engine


def main():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    print("PostgreSQL connection: OK")
    print("Tables:")

    for table in tables:
        print(f"  - {table}")


if __name__ == "__main__":
    main()