BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "books" (
	"id"	INTEGER,
	"title"	TEXT NOT NULL,
	"filename"	TEXT NOT NULL,
	"path"	TEXT NOT NULL UNIQUE,
	"slide_count"	INTEGER DEFAULT 0,
	"mtime"	REAL,
	"size"	INTEGER,
	"shamela_id"	TEXT,
	"folder"	TEXT,
	"part_label"	TEXT,
	"sort_num"	INTEGER DEFAULT 0,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "embeddings" (
	"slide_id"	INTEGER,
	"book_id"	INTEGER NOT NULL,
	"vector"	TEXT NOT NULL,
	"model"	TEXT,
	PRIMARY KEY("slide_id")
);
CREATE TABLE IF NOT EXISTS "headings" (
	"id"	INTEGER,
	"book_id"	INTEGER NOT NULL,
	"slide_id"	INTEGER NOT NULL,
	"level"	INTEGER NOT NULL,
	"text"	TEXT NOT NULL,
	"anchor_id"	TEXT NOT NULL DEFAULT '',
	"source"	TEXT NOT NULL DEFAULT 'inline',
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTSAL TABLE search_index USING fts5(
            normalized_text,
            book_id UNINDEXED,
            slide_id UNINDEXED,
            heading_text UNINDEXED,
            source UNINDEXED
        );
CREATE TABLE IF NOT EXISTS "search_index_config" (
	"k"	,
	"v"	,
	PRIMARY KEY("k")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "search_index_content" (
	"id"	INTEGER,
	"c0"	,
	"c1"	,
	"c2"	,
	"c3"	,
	"c4"	,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "search_index_data" (
	"id"	INTEGER,
	"block"	BLOB,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "search_index_docsize" (
	"id"	INTEGER,
	"sz"	BLOB,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "search_index_idx" (
	"segid"	,
	"term"	,
	"pgno"	,
	PRIMARY KEY("segid","term")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "slides" (
	"id"	INTEGER,
	"book_id"	INTEGER NOT NULL,
	"slide_number"	INTEGER NOT NULL,
	"html_content"	TEXT NOT NULL,
	"plain_text"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE INDEX IF NOT EXISTS "idx_embeddings_book" ON "embeddings" (
	"book_id"
);
CREATE INDEX IF NOT EXISTS "idx_headings_book" ON "headings" (
	"book_id"
);
CREATE INDEX IF NOT EXISTS "idx_slides_book" ON "slides" (
	"book_id"
);
CREATE INDEX IF NOT EXISTS "idx_slides_book_num" ON "slides" (
	"book_id",
	"slide_number"
);
COMMIT;
