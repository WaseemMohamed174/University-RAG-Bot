import logging
from collections import Counter

from qdrant_client import AsyncQdrantClient, models

from . import config

log = logging.getLogger(__name__)
_client: AsyncQdrantClient | None = None


# ============== الاتصال ==============
def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = (
            AsyncQdrantClient(location=":memory:")
            if config.QDRANT_URL == ":memory:"
            else AsyncQdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


# ============== الكوليكشن ==============
async def init_collection() -> None:
    c, name = client(), config.COLLECTION_NAME
    if not await c.collection_exists(name):
        await c.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
        )
        try:
            await c.create_payload_index(name, "source", models.PayloadSchemaType.KEYWORD)
        except Exception as e:
            log.debug("payload index skipped: %s", e)
        log.info("✅ تم إنشاء الكوليكشن %s (dim=%d)", name, config.EMBED_DIM)
        return
    size = getattr((await c.get_collection(name)).config.params.vectors, "size", None)
    if size and size != config.EMBED_DIM:
        raise RuntimeError(
            f"الكوليكشن '{name}' أبعادها {size} بس EMBED_DIM={config.EMBED_DIM}. "
            "غيّر COLLECTION_NAME أو امسح الكوليكشن القديمة وأعد رفع الملفات."
        )


# ============== الرفع والبحث والحذف ==============
async def upsert(points: list[models.PointStruct], batch: int = 64) -> None:
    for i in range(0, len(points), batch):
        await client().upsert(collection_name=config.COLLECTION_NAME, points=points[i : i + batch])


async def search(vector: list[float], limit: int) -> list[models.ScoredPoint]:
    res = await client().query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=limit,
        with_payload=True,
        score_threshold=config.MIN_SCORE or None,
    )
    return res.points


async def delete_source(source: str) -> None:
    condition = models.FieldCondition(key="source", match=models.MatchValue(value=source))
    await client().delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=models.Filter(must=[condition])),
    )


async def list_sources() -> Counter:
    counts: Counter = Counter()
    offset = None
    while True:
        points, offset = await client().scroll(
            collection_name=config.COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        counts.update((p.payload or {}).get("source", "?") for p in points)
        if offset is None:
            return counts
