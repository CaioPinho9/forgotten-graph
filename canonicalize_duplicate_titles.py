from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

from db import SQLITE_DB_PATH

DISAMBIGUATION_SUFFIX_RE = re.compile(r"\s+\(disambiguation\)$", re.IGNORECASE)
APOSTROPHE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "`": "'",
})

SIMPLE_PAGE_TITLE_TABLES = [
    "discovered_pages",
    "searched_pages",
    "failed_pages",
    "cluster_assignments",
    "nodes_layout",
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def normalize_whitespace(title: str) -> str:
    return " ".join((title or "").split())


def normalize_apostrophes(title: str) -> str:
    return normalize_whitespace(title).translate(APOSTROPHE_TRANSLATION)


def strip_disambiguation_suffix(title: str) -> str:
    return DISAMBIGUATION_SUFFIX_RE.sub("", normalize_apostrophes(title))


def normalized_family_key(title: str) -> str:
    return strip_disambiguation_suffix(title).casefold()


def canonical_preference_key(title: str) -> tuple[int, int, int, str]:
    normalized = normalize_apostrophes(title)
    stripped = strip_disambiguation_suffix(title)
    is_disambiguation = int(normalized != stripped)
    has_title_case = int(normalized == normalized.title())
    return (
        is_disambiguation,
        -has_title_case,
        len(stripped),
        normalized.casefold(),
    )


def fetch_all_titles(cur: sqlite3.Cursor) -> list[str]:
    tables_and_columns = [
        ("discovered_pages", "page_title"),
        ("searched_pages", "page_title"),
        ("failed_pages", "page_title"),
        ("pages", "page_title"),
        ("page_links", "source_title"),
        ("page_links", "target_title"),
        ("page_categories", "page_title"),
        ("cluster_assignments", "page_title"),
        ("nodes_layout", "page_title"),
    ]

    titles: set[str] = set()
    for table_name, column_name in tables_and_columns:
        cur.execute(f"SELECT DISTINCT {column_name} FROM {table_name}")
        for (title,) in cur.fetchall():
            if title:
                titles.add(title)
    return sorted(titles)


def build_alias_map(titles: list[str]) -> dict[str, str]:
    titles_by_family: dict[str, list[str]] = defaultdict(list)
    for title in titles:
        titles_by_family[normalized_family_key(title)].append(title)

    alias_map: dict[str, str] = {}

    for family_titles in titles_by_family.values():
        unique_titles = sorted(set(family_titles))
        if len(unique_titles) < 2:
            continue

        canonical_title = min(unique_titles, key=canonical_preference_key)
        exact_titles = set(unique_titles)
        exact_base_titles = {strip_disambiguation_suffix(title) for title in unique_titles}

        for title in unique_titles:
            if title == canonical_title:
                continue

            normalized_title = normalize_apostrophes(title)
            normalized_canonical = normalize_apostrophes(canonical_title)

            if normalized_title.casefold() == normalized_canonical.casefold():
                alias_map[title] = canonical_title
                continue

            stripped_title = strip_disambiguation_suffix(title)
            if (
                stripped_title in exact_titles
                and stripped_title in exact_base_titles
                and stripped_title != title
            ):
                alias_map[title] = stripped_title

    return alias_map


def create_temp_alias_map(cur: sqlite3.Cursor, alias_map: dict[str, str]) -> None:
    cur.execute("DROP TABLE IF EXISTS temp_alias_map")
    cur.execute(
        """
        CREATE TEMP TABLE temp_alias_map
        (
            old_title TEXT PRIMARY KEY,
            canonical_title TEXT NOT NULL
        )
        """
    )
    cur.executemany(
        """
        INSERT INTO temp_alias_map (old_title, canonical_title)
        VALUES (?, ?)
        """,
        sorted(alias_map.items()),
    )


def migrate_simple_page_title_table(cur: sqlite3.Cursor, table_name: str) -> None:
    cur.execute(f"DROP TABLE IF EXISTS temp_{table_name}_titles")
    if table_name in {"discovered_pages", "searched_pages"}:
        cur.execute(
            f"""
            CREATE TEMP TABLE temp_{table_name}_titles AS
            SELECT DISTINCT
                COALESCE(m.canonical_title, t.page_title) AS page_title
            FROM {table_name} t
            LEFT JOIN temp_alias_map m ON m.old_title = t.page_title
            """
        )
    elif table_name == "failed_pages":
        cur.execute(
            f"""
            CREATE TEMP TABLE temp_{table_name}_titles AS
            SELECT
                COALESCE(m.canonical_title, t.page_title) AS page_title,
                MIN(t.error_message) AS error_message
            FROM {table_name} t
            LEFT JOIN temp_alias_map m ON m.old_title = t.page_title
            GROUP BY COALESCE(m.canonical_title, t.page_title)
            """
        )
    elif table_name == "cluster_assignments":
        cur.execute(
            f"""
            CREATE TEMP TABLE temp_{table_name}_titles AS
            SELECT
                COALESCE(m.canonical_title, t.page_title) AS page_title,
                MIN(t.cluster_id) AS cluster_id
            FROM {table_name} t
            LEFT JOIN temp_alias_map m ON m.old_title = t.page_title
            GROUP BY COALESCE(m.canonical_title, t.page_title)
            """
        )
    elif table_name == "nodes_layout":
        cur.execute(
            f"""
            CREATE TEMP TABLE temp_{table_name}_titles AS
            SELECT
                canonical_page_title AS page_title,
                x,
                y,
                cluster_id,
                cluster_color,
                in_degree,
                out_degree,
                node_size
            FROM (
                SELECT
                    COALESCE(m.canonical_title, t.page_title) AS canonical_page_title,
                    t.x,
                    t.y,
                    t.cluster_id,
                    t.cluster_color,
                    t.in_degree,
                    t.out_degree,
                    t.node_size,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(m.canonical_title, t.page_title)
                        ORDER BY
                            t.node_size DESC,
                            t.out_degree DESC,
                            t.in_degree DESC,
                            t.page_title ASC
                    ) AS row_number
                FROM {table_name} t
                LEFT JOIN temp_alias_map m ON m.old_title = t.page_title
            )
            WHERE row_number = 1
            """
        )

    cur.execute(f"DELETE FROM {table_name}")

    if table_name in {"discovered_pages", "searched_pages"}:
        cur.execute(
            f"""
            INSERT INTO {table_name} (page_title)
            SELECT page_title
            FROM temp_{table_name}_titles
            """
        )
    elif table_name == "failed_pages":
        cur.execute(
            f"""
            INSERT INTO failed_pages (page_title, error_message)
            SELECT page_title, error_message
            FROM temp_{table_name}_titles
            """
        )
    elif table_name == "cluster_assignments":
        cur.execute(
            f"""
            INSERT INTO cluster_assignments (page_title, cluster_id)
            SELECT page_title, cluster_id
            FROM temp_{table_name}_titles
            """
        )
    elif table_name == "nodes_layout":
        cur.execute(
            f"""
            INSERT INTO nodes_layout (
                page_title, x, y, cluster_id, cluster_color, in_degree, out_degree, node_size
            )
            SELECT
                page_title, x, y, cluster_id, cluster_color, in_degree, out_degree, node_size
            FROM temp_{table_name}_titles
            """
        )

    cur.execute(f"DROP TABLE temp_{table_name}_titles")


def migrate_pages(cur: sqlite3.Cursor) -> None:
    cur.execute("DROP TABLE IF EXISTS temp_pages_canonical")
    cur.execute(
        """
        CREATE TEMP TABLE temp_pages_canonical AS
        SELECT
            COALESCE(m.canonical_title, p.page_title) AS page_title,
            MAX(COALESCE(p.links_out_count, 0)) AS links_out_count,
            MAX(COALESCE(p.categories_count, 0)) AS categories_count,
            MIN(p.error_message) AS error_message
        FROM pages p
        LEFT JOIN temp_alias_map m ON m.old_title = p.page_title
        GROUP BY COALESCE(m.canonical_title, p.page_title)
        """
    )
    cur.execute("DELETE FROM pages")
    cur.execute(
        """
        INSERT INTO pages (page_title, links_out_count, categories_count, error_message)
        SELECT page_title, links_out_count, categories_count, error_message
        FROM temp_pages_canonical
        """
    )
    cur.execute("DROP TABLE temp_pages_canonical")


def migrate_page_links(cur: sqlite3.Cursor) -> None:
    edges_before = cur.execute("SELECT COUNT(*) FROM page_links").fetchone()[0]
    nodes_before = cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT source_title AS page_title FROM page_links
            UNION
            SELECT target_title AS page_title FROM page_links
        )
        """
    ).fetchone()[0]

    cur.execute("DROP TABLE IF EXISTS temp_page_links_canonical")
    cur.execute(
        """
        CREATE TEMP TABLE temp_page_links_canonical AS
        SELECT DISTINCT
            COALESCE(ms.canonical_title, l.source_title) AS source_title,
            COALESCE(mt.canonical_title, l.target_title) AS target_title
        FROM page_links l
        LEFT JOIN temp_alias_map ms ON ms.old_title = l.source_title
        LEFT JOIN temp_alias_map mt ON mt.old_title = l.target_title
        """
    )
    cur.execute("DELETE FROM page_links")
    cur.execute(
        """
        INSERT INTO page_links (source_title, target_title)
        SELECT source_title, target_title
        FROM temp_page_links_canonical
        """
    )
    cur.execute("DROP TABLE temp_page_links_canonical")

    edges_after = cur.execute("SELECT COUNT(*) FROM page_links").fetchone()[0]
    nodes_after = cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT source_title AS page_title FROM page_links
            UNION
            SELECT target_title AS page_title FROM page_links
        )
        """
    ).fetchone()[0]

    print(f"page_links edges before: {edges_before}")
    print(f"page_links edges after: {edges_after}")
    print(f"page_links nodes before: {nodes_before}")
    print(f"page_links nodes after: {nodes_after}")


def migrate_page_categories(cur: sqlite3.Cursor) -> None:
    cur.execute("DROP TABLE IF EXISTS temp_page_categories_canonical")
    cur.execute(
        """
        CREATE TEMP TABLE temp_page_categories_canonical AS
        SELECT DISTINCT
            COALESCE(m.canonical_title, c.page_title) AS page_title,
            c.category_name AS category_name
        FROM page_categories c
        LEFT JOIN temp_alias_map m ON m.old_title = c.page_title
        """
    )
    cur.execute("DELETE FROM page_categories")
    cur.execute(
        """
        INSERT INTO page_categories (page_title, category_name)
        SELECT page_title, category_name
        FROM temp_page_categories_canonical
        """
    )
    cur.execute("DROP TABLE temp_page_categories_canonical")


def backfill_pages_and_discovered_pages_from_graph(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        INSERT OR IGNORE INTO pages (page_title, links_out_count, categories_count, error_message)
        SELECT node_title, 0, 0, NULL
        FROM (
            SELECT source_title AS node_title FROM page_links
            UNION
            SELECT target_title AS node_title FROM page_links
        )
        """
    )
    cur.execute(
        """
        INSERT OR IGNORE INTO discovered_pages (page_title)
        SELECT page_title FROM pages
        """
    )
    cur.execute(
        """
        INSERT OR IGNORE INTO discovered_pages (page_title)
        SELECT node_title
        FROM (
            SELECT source_title AS node_title FROM page_links
            UNION
            SELECT target_title AS node_title FROM page_links
        )
        """
    )


def print_mapping_sample(alias_map: dict[str, str], limit: int = 20) -> None:
    print("sample mappings:")
    for old_title, canonical_title in sorted(alias_map.items())[:limit]:
        print(f"  {old_title} -> {canonical_title}")


def run_migration() -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        all_titles = fetch_all_titles(cur)
        alias_map = build_alias_map(all_titles)

        print(f"total titles scanned: {len(all_titles)}")
        print(f"total duplicates found: {len(alias_map)}")
        if alias_map:
            print_mapping_sample(alias_map)

        cur.execute("BEGIN")

        if alias_map:
            create_temp_alias_map(cur, alias_map)
            for table_name in SIMPLE_PAGE_TITLE_TABLES:
                migrate_simple_page_title_table(cur, table_name)
            migrate_pages(cur)
            migrate_page_links(cur)
            migrate_page_categories(cur)
            cur.execute("DROP TABLE IF EXISTS temp_alias_map")
        else:
            print("sample mappings:")

        backfill_pages_and_discovered_pages_from_graph(cur)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()
