"""
Fashion category hierarchy for better query understanding.
Helps system understand product relationships and categories.

Used for:
  - Query expansion with related categories
  - Better semantic understanding
  - Filtering by fashion type
"""

FASHION_TAXONOMY = {
    "tops": {
        "shirt": ["casual shirt", "dress shirt", "formal shirt", "button-up"],
        "tshirt": ["casual tshirt", "graphic tee", "plain tshirt", "t-shirt"],
        "blouse": ["silk blouse", "cotton blouse", "formal blouse"],
        "sweater": ["wool sweater", "cardigan", "pullover", "hoodie"],
        "crop": ["crop top", "bralette", "bustier"],
    },
    "bottoms": {
        "jeans": ["blue jeans", "black jeans", "skinny jeans", "denim"],
        "saree": ["silk saree", "cotton saree", "party saree"],
        "dress": ["casual dress", "formal dress", "party dress", "evening gown"],
        "shorts": ["denim shorts", "cotton shorts", "bermuda shorts"],
        "leggings": ["yoga leggings", "cotton leggings", "printed leggings"],
    },
    "outerwear": {
        "jacket": ["leather jacket", "denim jacket", "blazer", "sport coat"],
        "coat": ["winter coat", "wool coat", "parka"],
        "shrug": ["silk shrug", "cotton shrug"],
    },
    "ethnic": {
        "kurta": ["cotton kurta", "silk kurta", "designer kurta"],
        "kurti": ["casual kurti", "printed kurti", "embroidered kurti"],
        "lehenga": ["wedding lehenga", "party lehenga", "silk lehenga"],
        "salwar": ["salwar suit", "salwar kameez"],
    },
    "activewear": {
        "sports": ["sports bra", "yoga pants", "gym wear"],
        "trainers": ["running shoes", "gym shoes", "sneakers"],
    },
}

# Similarity scores between different fashion items
CATEGORY_SIMILARITY = {
    # Similar products (high score)
    ("shirt", "blouse"): 0.8,
    ("tshirt", "shirt"): 0.7,
    ("jeans", "shorts"): 0.7,
    ("dress", "saree"): 0.6,
    ("jacket", "coat"): 0.8,
    ("kurta", "kurti"): 0.8,
    ("lehenga", "saree"): 0.7,
    
    # Somewhat related (medium score)
    ("tshirt", "sweater"): 0.5,
    ("jeans", "dress"): 0.4,
    ("jacket", "cardigan"): 0.6,
    
    # Dissimilar products (low score)
    ("shirt", "saree"): 0.2,
    ("tshirt", "lehenga"): 0.1,
    ("jeans", "blouse"): 0.2,
}


def get_category_for_item(item_name: str) -> tuple[str, str] | None:
    """
    Find the category and subcategory for a given item.
    
    Args:
        item_name: Name of the fashion item
    
    Returns:
        Tuple of (main_category, subcategory) or None if not found
    """
    item_lower = item_name.lower()
    for main_cat, subcats in FASHION_TAXONOMY.items():
        for subcat, variations in subcats.items():
            if any(v in item_lower for v in variations):
                return (main_cat, subcat)
    return None


def get_similar_items(category: str) -> list[str]:
    """
    Get similar fashion items in the same category family.
    
    Args:
        category: Fashion subcategory (e.g., "shirt")
    
    Returns:
        List of similar item types
    """
    category_lower = category.lower()
    for main_cat, subcats in FASHION_TAXONOMY.items():
        if category_lower in subcats:
            # Return other items from same main category
            other_items = [
                subcat for subcat in subcats.keys() 
                if subcat != category_lower
            ]
            return other_items
    return []


def boost_query_with_taxonomy(query: str) -> str:
    """
    Boost query by adding related category synonyms.
    
    Example:
        Input:  "casual shirt"
        Output: "casual shirt dress shirt formal shirt blouse sweater"
                (added related items)
    
    Args:
        query: Original search query
    
    Returns:
        Expanded query with related terms
    """
    expanded = query
    query_lower = query.lower()
    
    # Find detected categories
    for main_cat, subcats in FASHION_TAXONOMY.items():
        for subcat, variations in subcats.items():
            if any(v in query_lower for v in variations):
                # Found a matching category
                # Add related items from same family
                related = get_similar_items(subcat)
                if related:
                    expanded += " " + " ".join(related)
                break
    
    return expanded.strip()


def get_attribute_keywords(query: str) -> dict[str, list[str]]:
    """
    Extract fashion attributes from query.
    
    Args:
        query: Search query
    
    Returns:
        Dictionary with attribute types and values found
    """
    query_lower = query.lower()
    attributes = {
        "colors": [],
        "materials": [],
        "occasions": [],
        "styles": [],
    }
    
    # Color keywords
    colors = ["red", "black", "white", "blue", "green", "yellow", "pink", 
              "gold", "silver", "navy", "beige", "brown", "gray", "purple"]
    for color in colors:
        if color in query_lower:
            attributes["colors"].append(color)
    
    # Material keywords
    materials = ["silk", "cotton", "polyester", "wool", "linen", "satin", 
                 "denim", "velvet", "chiffon", "georgette"]
    for material in materials:
        if material in query_lower:
            attributes["materials"].append(material)
    
    # Occasion keywords
    occasions = ["casual", "formal", "party", "wedding", "office", "sports", 
                 "beach", "date night", "evening"]
    for occasion in occasions:
        if occasion in query_lower:
            attributes["occasions"].append(occasion)
    
    # Style keywords
    styles = ["slim", "regular", "oversized", "fitted", "loose", "short", 
              "midi", "long", "vintage", "modern", "ethnic"]
    for style in styles:
        if style in query_lower:
            attributes["styles"].append(style)
    
    # Remove empty keys
    return {k: v for k, v in attributes.items() if v}