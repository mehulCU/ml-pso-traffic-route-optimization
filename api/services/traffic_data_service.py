from api.database import get_db_connection


def get_observation_count():
    """Return the total number of stored traffic observations."""
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM traffic_observations"
            )
            return cursor.fetchone()[0]
    finally:
        connection.close()


def get_recent_observations(limit=10):
    """Return the most recent traffic observations."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    connection = get_db_connection()

    try:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT
                    segment_id,
                    timestamp,
                    street_name,
                    speed_limit,
                    probe_count
                FROM traffic_observations
                ORDER BY timestamp DESC, segment_id
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()
    finally:
        connection.close()
