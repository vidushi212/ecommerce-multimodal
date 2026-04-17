import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from models.text_retrieval import TextRetrieval

# Initialize model
model = TextRetrieval()
model.initialize()

# Search
query = "casual tshirt"
results = model.search(query, top_k=20)

# Print IDs
print(f"\n🔍 Query: '{query}'")
print(f"📊 Top 20 Results:")
print("-" * 50)

retrieved_ids = []
for i, product in enumerate(results, 1):
    product_id = product["id"]
    title = product["title"]
    score = product.get("score", 0)
    retrieved_ids.append(product_id)
    print(f"{i:2d}. ID: {product_id:6s} | Score: {score:.3f} | {title[:40]}")

print("\n" + "=" * 50)
print("Comma-separated IDs for Evaluate:")
print(",".join(retrieved_ids))