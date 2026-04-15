from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB_PATH = DB_DIR / "forgotten_graph.sqlite3"


def get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def cursor():
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS discovered_pages
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            discovered_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS searched_pages
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            searched_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS failed_pages
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            failed_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP,
            error_message
            TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pages
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            links_out_count
            INT,
            categories_count
            INT,
            error_message
            TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS page_links
        (
            source_title
            TEXT
            NOT
            NULL,
            target_title
            TEXT
            NOT
            NULL,
            PRIMARY
            KEY
        (
            source_title,
            target_title
        )
            )
        """,
        """
        CREATE TABLE IF NOT EXISTS page_categories
        (
            page_title
            TEXT
            NOT
            NULL,
            category_name
            TEXT
            NOT
            NULL,
            PRIMARY
            KEY
        (
            page_title,
            category_name
        )
            )
        """,
        """
        CREATE TABLE IF NOT EXISTS cluster_assignments
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            cluster_id
            INT
            NOT
            NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cluster_colors
        (
            cluster_id
            INT
            PRIMARY
            KEY,
            cluster_color
            TEXT
            NOT
            NULL,
            cluster_name
            TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS nodes_layout
        (
            page_title
            TEXT
            PRIMARY
            KEY,
            x
            REAL
            NOT
            NULL,
            y
            REAL
            NOT
            NULL,
            cluster_id
            INT
            NOT
            NULL,
            cluster_color
            TEXT
            NOT
            NULL,
            in_degree
            INT
            NOT
            NULL,
            out_degree
            INT
            NOT
            NULL,
            node_size
            REAL
            NOT
            NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_page_links_source ON page_links(source_title)",
        "CREATE INDEX IF NOT EXISTS idx_page_links_target ON page_links(target_title)",
        "CREATE INDEX IF NOT EXISTS idx_page_categories_page ON page_categories(page_title)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_layout_xy ON nodes_layout(x, y)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_layout_cluster ON nodes_layout(cluster_id)",
    ]

    with cursor() as (_, cur):
        for stmt in ddl:
            cur.execute(stmt)

    migrate_cluster_colors_table()
    cleanup_stale_failed_pages()


def migrate_cluster_colors_table() -> None:
    with cursor() as (_, cur):
        cur.execute("PRAGMA table_info(cluster_colors)")
        columns = {row[1] for row in cur.fetchall()}

        if "cluster_name" not in columns:
            cur.execute("ALTER TABLE cluster_colors ADD COLUMN cluster_name TEXT")


def cleanup_stale_failed_pages() -> None:
    with cursor() as (_, cur):
        cur.execute(
            """
            DELETE
            FROM failed_pages
            WHERE page_title IN (SELECT f.page_title
                                 FROM failed_pages f
                                          JOIN searched_pages s ON s.page_title = f.page_title)
            """
        )


def insert_discovered_titles(titles: list[str]) -> None:
    if not titles:
        return
    with cursor() as (_, cur):
        cur.executemany(
            "INSERT OR IGNORE INTO discovered_pages (page_title) VALUES (?)",
            [(t,) for t in titles],
        )


def load_directed_node_adjacency(discovered_only: bool = True) -> dict[str, dict[str, list[str]]]:
    with cursor() as (_, cur):
        if discovered_only:
            cur.execute(
                """
                SELECT l.source_title, l.target_title
                FROM page_links l
                         JOIN discovered_pages ds ON ds.page_title = l.source_title
                         JOIN discovered_pages dt ON dt.page_title = l.target_title
                """
            )
        else:
            cur.execute(
                """
                SELECT source_title, target_title
                FROM page_links
                """
            )

        rows = cur.fetchall()

    adjacency: dict[str, dict[str, set[str]]] = {}

    for source, target in rows:
        adjacency.setdefault(source, {"out": set(), "in": set()})
        adjacency.setdefault(target, {"out": set(), "in": set()})
        adjacency[source]["out"].add(target)
        adjacency[target]["in"].add(source)

    return {
        node_id: {
            "out": sorted(data["out"]),
            "in": sorted(data["in"]),
        }
        for node_id, data in adjacency.items()
    }


def load_node_adjacency(discovered_only: bool = True) -> dict[str, list[str]]:
    with cursor() as (_, cur):
        if discovered_only:
            cur.execute(
                """
                SELECT l.source_title, l.target_title
                FROM page_links l
                         JOIN discovered_pages ds ON ds.page_title = l.source_title
                         JOIN discovered_pages dt ON dt.page_title = l.target_title
                """
            )
        else:
            cur.execute(
                """
                SELECT source_title, target_title
                FROM page_links
                """
            )

        rows = cur.fetchall()

    adjacency: dict[str, set[str]] = {}

    for source, target in rows:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    return {
        node_id: sorted(neighbors)
        for node_id, neighbors in adjacency.items()
    }


def clear_cluster_and_layout_data() -> None:
    with cursor() as (_, cur):
        cur.execute("DELETE FROM cluster_assignments")
        cur.execute("DELETE FROM cluster_colors")
        cur.execute("DELETE FROM nodes_layout")


def count_discovered_titles() -> int:
    with cursor() as (_, cur):
        cur.execute("SELECT COUNT(*) FROM discovered_pages")
        return int(cur.fetchone()[0])


def load_discovered_titles() -> list[str]:
    with cursor() as (_, cur):
        cur.execute("SELECT page_title FROM discovered_pages ORDER BY page_title")
        return [row[0] for row in cur.fetchall()]


def load_discovered_title_set() -> set[str]:
    return set(load_discovered_titles())


def load_searched_title_set() -> set[str]:
    with cursor() as (_, cur):
        cur.execute("SELECT page_title FROM searched_pages")
        return {row[0] for row in cur.fetchall()}


def load_failed_title_set() -> set[str]:
    with cursor() as (_, cur):
        cur.execute("SELECT page_title FROM failed_pages")
        return {row[0] for row in cur.fetchall()}


def store_crawl_results(records: list[dict]) -> None:
    """
    Each record:
      {
        page_title, links_out, categories, error
      }
    """
    if not records:
        return

    with cursor() as (_, cur):
        for r in records:
            title = r["page_title"]
            links = r.get("links_out", [])
            categories = r.get("categories", [])
            error = r.get("error")

            cur.execute(
                """
                INSERT INTO pages (page_title, links_out_count, categories_count, error_message)
                VALUES (?, ?, ?, ?) ON CONFLICT(page_title) DO
                UPDATE SET
                    links_out_count = excluded.links_out_count,
                    categories_count = excluded.categories_count,
                    error_message = excluded.error_message
                """,
                (title, len(links), len(categories), error),
            )

            if error is None:
                cur.execute(
                    "INSERT OR IGNORE INTO searched_pages (page_title) VALUES (?)",
                    (title,),
                )
                cur.execute(
                    "DELETE FROM failed_pages WHERE page_title = ?",
                    (title,),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO failed_pages (page_title, error_message)
                    VALUES (?, ?) ON CONFLICT(page_title) DO
                    UPDATE SET
                        error_message = excluded.error_message,
                        failed_at = CURRENT_TIMESTAMP
                    """,
                    (title, error),
                )

            if error is None:
                for target in links:
                    cur.execute(
                        """
                        INSERT
                        OR IGNORE INTO page_links (source_title, target_title)
                        VALUES (?, ?)
                        """,
                        (title, target),
                    )

                for category in categories:
                    cur.execute(
                        """
                        INSERT
                        OR IGNORE INTO page_categories (page_title, category_name)
                        VALUES (?, ?)
                        """,
                        (title, category),
                    )


def read_filtered_edges(discovered_only: bool = True) -> tuple[list[str], list[tuple[str, str]]]:
    with cursor() as (_, cur):
        if discovered_only:
            cur.execute(
                """
                SELECT l.source_title, l.target_title
                FROM page_links l
                         JOIN discovered_pages ds ON ds.page_title = l.source_title
                         JOIN discovered_pages dt ON dt.page_title = l.target_title
                """
            )
        else:
            cur.execute("SELECT source_title, target_title FROM page_links")

        rows = cur.fetchall()

    edges = [(row[0], row[1]) for row in rows]
    nodes = sorted({s for s, _ in edges} | {t for _, t in edges})
    return nodes, edges


def load_page_categories() -> dict[str, list[str]]:
    with cursor() as (_, cur):
        cur.execute(
            """
            SELECT page_title, category_name
            FROM page_categories
            ORDER BY page_title, category_name
            """
        )
        rows = cur.fetchall()

    categories: dict[str, list[str]] = {}
    for page_title, category_name in rows:
        title = (page_title or "").strip()
        category = (category_name or "").strip()
        if not title or not category:
            continue
        categories.setdefault(title, []).append(category)
    return categories


def save_cluster_assignments(
        assignments: list[tuple[str, int]],
        cluster_metadata: dict[int, dict[str, str | None]],
) -> None:
    with cursor() as (_, cur):
        for title, cluster_id in assignments:
            cur.execute(
                """
                INSERT INTO cluster_assignments (page_title, cluster_id)
                VALUES (?, ?) ON CONFLICT(page_title) DO
                UPDATE SET
                    cluster_id = excluded.cluster_id
                """,
                (title, cluster_id),
            )

        for cluster_id, metadata in cluster_metadata.items():
            cluster_color = metadata["cluster_color"]
            cluster_name = metadata.get("cluster_name")
            cur.execute(
                """
                INSERT INTO cluster_colors (cluster_id, cluster_color, cluster_name)
                VALUES (?, ?, ?) ON CONFLICT(cluster_id) DO
                UPDATE SET
                    cluster_color = excluded.cluster_color
                    , cluster_name = excluded.cluster_name
                """,
                (cluster_id, cluster_color, cluster_name),
            )


def load_cluster_assignments() -> dict[str, int]:
    with cursor() as (_, cur):
        cur.execute("SELECT page_title, cluster_id FROM cluster_assignments")
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def load_cluster_colors() -> dict[int, str]:
    metadata = load_cluster_metadata()
    return {
        cluster_id: cluster_info["cluster_color"]
        for cluster_id, cluster_info in metadata.items()
    }


def load_cluster_metadata() -> dict[int, dict[str, str | None]]:
    with cursor() as (_, cur):
        cur.execute(
            """
            SELECT cluster_id, cluster_color, cluster_name
            FROM cluster_colors
            """
        )
        rows = cur.fetchall()

    return {
        int(row[0]): {
            "cluster_color": row[1],
            "cluster_name": row[2],
        }
        for row in rows
    }


def save_nodes_layout(rows: list[tuple[str, float, float, int, str, int, int, float]]) -> None:
    with cursor() as (_, cur):
        for row in rows:
            cur.execute(
                """
                INSERT INTO nodes_layout
                    (page_title, x, y, cluster_id, cluster_color, in_degree, out_degree, node_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(page_title) DO
                UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    cluster_id = excluded.cluster_id,
                    cluster_color = excluded.cluster_color,
                    in_degree = excluded.in_degree,
                    out_degree = excluded.out_degree,
                    node_size = excluded.node_size
                """,
                row,
            )


def load_nodes_layout() -> list[dict]:
    with cursor() as (_, cur):
        cur.execute(
            """
            SELECT page_title,
                   x,
                   y,
                   cluster_id,
                   cluster_color,
                   in_degree,
                   out_degree,
                   node_size
            FROM nodes_layout
            """
        )
        rows = cur.fetchall()

    return [
        {
            "page_title": r[0],
            "x": float(r[1]),
            "y": float(r[2]),
            "cluster_id": int(r[3]),
            "cluster_color": r[4],
            "in_degree": int(r[5]),
            "out_degree": int(r[6]),
            "node_size": float(r[7]),
        }
        for r in rows
    ]


def load_edges_for_chunks(discovered_only: bool = True) -> list[tuple[str, str]]:
    return read_filtered_edges(discovered_only=discovered_only)[1]


def load_nodes_edges_by_title(title: str) -> list[str]:
    with cursor() as (_, cur):
        cur.execute(
            "SELECT target_title FROM page_links WHERE page_title = ?", (title,)
        )

    return cur.fetchall()
