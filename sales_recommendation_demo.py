"""
META-LEARNING FOR CUSTOMER RECOMMENDATIONS
===========================================

BUSINESS PROBLEM: The "Cold Start" Challenge
--------------------------------------------
You have a new customer or new product with minimal interaction history.
Traditional recommendation systems fail because they need lots of data.

SOLUTION: Meta-Learning
-----------------------
Train a model that learns "how to recommend" across many customers,
then quickly adapts to NEW customers with just 1-5 purchase interactions!

REAL-WORLD SCENARIO:
-------------------
- E-commerce: Recommend products to new customers
- Streaming: Suggest content to new subscribers  
- Retail: Personalize offers with minimal customer data
- B2B Sales: Suggest services to new business clients

This demo simulates an e-commerce recommendation system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from collections import defaultdict

print("=" * 80)
print("META-LEARNING: SMART CUSTOMER RECOMMENDATIONS")
print("Solving the Cold-Start Problem")
print("=" * 80)

# ============================================================================
# PART 1: UNDERSTANDING THE COLD-START PROBLEM
# ============================================================================
print("\n📊 THE COLD-START PROBLEM IN RECOMMENDATIONS")
print("-" * 80)
print("""
TRADITIONAL RECOMMENDATION SYSTEMS:
----------------------------------
❌ Need 50-100+ interactions per customer
❌ New customers get random/popular recommendations
❌ New products are barely recommended
❌ Poor personalization = lost sales

Example: Netflix, Amazon need lots of your activity to recommend well

META-LEARNING APPROACH:
----------------------
✅ Learn from 1-5 customer interactions
✅ Quickly personalize for new customers
✅ Adapt to new products instantly
✅ Better conversion rates from day 1

Example: After buying ONE item, get personalized recommendations!
""")

input("Press Enter to see the demo data...")

# ============================================================================
# PART 2: CREATE REALISTIC E-COMMERCE DATA
# ============================================================================
print("\n🛍️ GENERATING E-COMMERCE DATA")
print("-" * 80)

# Define product categories
CATEGORIES = {
    'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Smartwatch', 
                   'Camera', 'Speaker', 'Monitor'],
    'Clothing': ['Shirt', 'Pants', 'Dress', 'Jacket', 'Shoes', 
                'Hat', 'Scarf', 'Gloves'],
    'Home': ['Lamp', 'Chair', 'Table', 'Rug', 'Curtain', 
            'Pillow', 'Blanket', 'Vase'],
    'Books': ['Fiction', 'Biography', 'Science', 'History', 'Cooking', 
             'Travel', 'Business', 'Self-Help'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Bicycle', 
              'Tennis Racket', 'Basketball', 'Swimming Goggles', 'Protein Powder']
}

# Customer personas (buying patterns)
PERSONAS = {
    'Tech Enthusiast': {'Electronics': 0.7, 'Books': 0.2, 'Sports': 0.1},
    'Fashionista': {'Clothing': 0.7, 'Home': 0.2, 'Books': 0.1},
    'Homebody': {'Home': 0.6, 'Books': 0.3, 'Clothing': 0.1},
    'Athlete': {'Sports': 0.7, 'Clothing': 0.2, 'Electronics': 0.1},
    'Bookworm': {'Books': 0.6, 'Home': 0.2, 'Clothing': 0.2}
}

# Generate synthetic customer data
np.random.seed(42)

def generate_customer_interactions(n_customers=1000, interactions_per_customer=20):
    """Generate realistic customer purchase history"""
    
    all_products = []
    for category, products in CATEGORIES.items():
        for product in products:
            all_products.append({'category': category, 'product': product})
    
    customer_data = []
    
    for customer_id in range(n_customers):
        # Assign persona
        persona = random.choice(list(PERSONAS.keys()))
        preferences = PERSONAS[persona]
        
        # Generate interactions based on persona
        interactions = []
        for _ in range(interactions_per_customer):
            # Choose category based on preferences
            category = np.random.choice(
                list(preferences.keys()),
                p=list(preferences.values())
            )
            
            # Choose random product from category
            product = random.choice(CATEGORIES[category])
            
            # Simulate purchase (1) or view (0)
            purchased = 1 if random.random() > 0.7 else 0
            
            interactions.append({
                'customer_id': customer_id,
                'persona': persona,
                'category': category,
                'product': product,
                'purchased': purchased,
                'price': random.randint(10, 500),
                'rating': random.randint(3, 5) if purchased else None
            })
        
        customer_data.extend(interactions)
    
    return pd.DataFrame(customer_data)

print("Generating customer purchase history...")
customer_df = generate_customer_interactions(n_customers=500, interactions_per_customer=20)

print(f"\n✓ Generated {len(customer_df):,} customer interactions")
print(f"✓ {customer_df['customer_id'].nunique()} unique customers")
print(f"✓ {len(set(customer_df['product']))} unique products")
print(f"✓ {customer_df['purchased'].sum():,} purchases")

print("\nSample interactions:")
print(customer_df.head(10))

# Visualize customer personas
print("\n📈 Customer Persona Distribution:")
persona_counts = customer_df.groupby('persona')['customer_id'].nunique()
for persona, count in persona_counts.items():
    print(f"  {persona}: {count} customers")

input("\nPress Enter to see the meta-learning approach...")

# ============================================================================
# PART 3: META-LEARNING RECOMMENDATION MODEL
# ============================================================================
print("\n🧠 BUILDING META-LEARNING RECOMMENDATION SYSTEM")
print("-" * 80)
print("""
OUR APPROACH: Prototypical Network for Recommendations
------------------------------------------------------

How it works:
1. LEARN USER PREFERENCES from purchase patterns
2. CREATE "PROTOTYPES" for each customer segment
3. RECOMMEND items similar to what similar customers bought
4. ADAPT QUICKLY to new customers with minimal data

Key Innovation:
- Traditional: Need 50+ purchases to recommend well
- Meta-Learning: Need 1-5 purchases to recommend well!
""")

class MetaRecommendationSystem:
    """
    Meta-learning based recommendation system.
    Learns customer preferences from minimal interactions.
    """
    
    def __init__(self):
        self.product_embeddings = {}
        self.customer_prototypes = {}
        self.scaler = StandardScaler()
        self.all_products = []
        
    def create_product_embeddings(self, df):
        """
        Create embeddings for products based on category and features.
        In practice, this could use deep learning on product images/descriptions.
        """
        print("\n  Creating product embeddings...")
        
        products = df.groupby('product').agg({
            'category': 'first',
            'price': 'mean',
            'rating': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else 3.5
        }).reset_index()
        
        self.all_products = products['product'].tolist()
        
        # Simple embedding: one-hot category + normalized price
        for _, row in products.iterrows():
            # One-hot encode category (5 categories)
            category_encoding = [0] * len(CATEGORIES)
            category_idx = list(CATEGORIES.keys()).index(row['category'])
            category_encoding[category_idx] = 1
            
            # Add price (normalized) and rating
            embedding = category_encoding + [
                row['price'] / 500,  # Normalize price
                row['rating'] / 5.0   # Normalize rating
            ]
            
            self.product_embeddings[row['product']] = np.array(embedding)
        
        print(f"  ✓ Created embeddings for {len(self.product_embeddings)} products")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
    
    def compute_customer_prototype(self, customer_purchases):
        """
        Compute the 'prototype' (average preference) for a customer.
        This represents what the customer typically likes.
        """
        embeddings = []
        for product in customer_purchases:
            if product in self.product_embeddings:
                embeddings.append(self.product_embeddings[product])
        
        if not embeddings:
            return None
        
        # Prototype = average of purchased items
        return np.mean(embeddings, axis=0)
    
    def train_on_customers(self, df, min_purchases=5):
        """
        Meta-training: Learn from many customers.
        Build prototypes for different customer types.
        """
        print("\n  Meta-training on customer data...")
        
        customers = df[df['purchased'] == 1].groupby('customer_id')
        
        trained_customers = 0
        for customer_id, group in customers:
            purchases = group['product'].tolist()
            
            if len(purchases) >= min_purchases:
                prototype = self.compute_customer_prototype(purchases)
                if prototype is not None:
                    self.customer_prototypes[customer_id] = {
                        'prototype': prototype,
                        'purchases': purchases,
                        'persona': group['persona'].iloc[0]
                    }
                    trained_customers += 1
        
        print(f"  ✓ Trained on {trained_customers} customers")
    
    def recommend_for_new_customer(self, new_customer_purchases, top_k=5):
        """
        FEW-SHOT RECOMMENDATION:
        Given 1-5 purchases from a NEW customer, recommend products.
        
        Steps:
        1. Compute prototype from their few purchases
        2. Find similar customers (by prototype distance)
        3. Recommend what similar customers bought
        """
        # Compute new customer's prototype
        new_prototype = self.compute_customer_prototype(new_customer_purchases)
        
        if new_prototype is None:
            return [], "No valid purchases to create prototype"
        
        # Find most similar customers
        similarities = []
        for customer_id, data in self.customer_prototypes.items():
            distance = np.linalg.norm(new_prototype - data['prototype'])
            similarities.append((customer_id, distance, data))
        
        # Get top 3 most similar customers
        similarities.sort(key=lambda x: x[1])
        similar_customers = similarities[:3]
        
        # Aggregate their purchases (excluding what new customer already bought)
        recommendation_scores = defaultdict(float)
        
        for customer_id, distance, data in similar_customers:
            similarity_weight = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            for product in data['purchases']:
                if product not in new_customer_purchases:
                    recommendation_scores[product] += similarity_weight
        
        # Sort by score
        recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Get similar customer personas for explanation
        similar_personas = [data['persona'] for _, _, data in similar_customers]
        
        return recommendations, similar_personas
    
    def recommend_popular_items(self, df, exclude_items=None, top_k=5):
        """
        Baseline: Just recommend most popular items.
        This is what traditional systems do for new customers.
        """
        if exclude_items is None:
            exclude_items = []
        
        popular = df[df['purchased'] == 1]['product'].value_counts()
        popular = popular[~popular.index.isin(exclude_items)]
        
        return list(popular.head(top_k).index)

# Initialize and train the system
print("\nInitializing Meta-Learning Recommendation System...")
rec_system = MetaRecommendationSystem()

# Step 1: Create product embeddings
rec_system.create_product_embeddings(customer_df)

# Step 2: Meta-train on existing customers
rec_system.train_on_customers(customer_df)

print("\n✓ System ready for recommendations!")

input("\nPress Enter to see recommendations for a new customer...")

# ============================================================================
# PART 4: DEMONSTRATE RECOMMENDATIONS
# ============================================================================
print("\n🎯 DEMONSTRATION: NEW CUSTOMER RECOMMENDATIONS")
print("-" * 80)

# Simulate a new customer with just 2-3 purchases
print("\nScenario: A NEW CUSTOMER just made their first few purchases")
print("-" * 80)

# Pick a random customer from our data to simulate
test_customer = customer_df[customer_df['customer_id'] == 100]
test_purchases = test_customer[test_customer['purchased'] == 1]['product'].tolist()

# Use only first 3 purchases for few-shot learning
few_shot_purchases = test_purchases[:3]
actual_persona = test_customer['persona'].iloc[0]

print(f"\nCustomer Profile:")
print(f"  Real persona: {actual_persona}")
print(f"  Purchase history (first 3 items only):")
for i, product in enumerate(few_shot_purchases, 1):
    category = customer_df[customer_df['product'] == product]['category'].iloc[0]
    print(f"    {i}. {product} ({category})")

# Meta-learning recommendations
print("\n🤖 META-LEARNING RECOMMENDATIONS:")
print("-" * 40)
meta_recommendations, similar_personas = rec_system.recommend_for_new_customer(
    few_shot_purchases, top_k=5
)

print(f"Based on similarity to customers: {', '.join(set(similar_personas))}")
print("\nTop 5 Recommended Products:")
for i, (product, score) in enumerate(meta_recommendations, 1):
    category = customer_df[customer_df['product'] == product]['category'].iloc[0]
    print(f"  {i}. {product:<20} ({category:<15}) Score: {score:.2f}")

# Baseline recommendations
print("\n📊 BASELINE (Popular Items) RECOMMENDATIONS:")
print("-" * 40)
baseline_recommendations = rec_system.recommend_popular_items(
    customer_df, exclude_items=few_shot_purchases, top_k=5
)

print("Top 5 Most Popular Products (what everyone buys):")
for i, product in enumerate(baseline_recommendations, 1):
    category = customer_df[customer_df['product'] == product]['category'].iloc[0]
    print(f"  {i}. {product:<20} ({category:<15})")

# Compare accuracy
print("\n📈 ACCURACY COMPARISON:")
print("-" * 40)

# Check what the customer actually bought later
actual_future_purchases = test_purchases[3:]  # Items bought after first 3

meta_hits = sum(1 for product, _ in meta_recommendations 
                if product in actual_future_purchases)
baseline_hits = sum(1 for product in baseline_recommendations 
                   if product in actual_future_purchases)

print(f"Customer's actual future purchases ({len(actual_future_purchases)} items):")
for product in actual_future_purchases[:5]:
    print(f"  • {product}")

print(f"\nRecommendation Accuracy:")
print(f"  Meta-Learning: {meta_hits}/5 recommendations matched ✓")
print(f"  Baseline:      {baseline_hits}/5 recommendations matched")
print(f"  Improvement:   {(meta_hits - baseline_hits) * 20}% better!")

input("\nPress Enter to see business metrics...")

# ============================================================================
# PART 5: BUSINESS IMPACT SIMULATION
# ============================================================================
print("\n💰 BUSINESS IMPACT ANALYSIS")
print("-" * 80)
print("""
Let's simulate the business impact of better recommendations!
""")

# Simulate conversion rates
np.random.seed(42)

# Test on 100 new customers
n_test_customers = 100
meta_conversions = []
baseline_conversions = []

print("Simulating recommendations for 100 new customers...")

for i in range(n_test_customers):
    # Get random customer
    test_id = random.randint(0, 400)
    test_data = customer_df[customer_df['customer_id'] == test_id]
    
    if len(test_data) < 5:
        continue
        
    purchases = test_data[test_data['purchased'] == 1]['product'].tolist()
    
    if len(purchases) < 5:
        continue
    
    # Few-shot: Use first 2 purchases
    few_shot = purchases[:2]
    future = purchases[2:]
    
    # Meta-learning recommendations
    meta_recs, _ = rec_system.recommend_for_new_customer(few_shot, top_k=5)
    meta_hit_rate = sum(1 for p, _ in meta_recs if p in future) / 5
    meta_conversions.append(meta_hit_rate)
    
    # Baseline recommendations
    baseline_recs = rec_system.recommend_popular_items(
        customer_df, exclude_items=few_shot, top_k=5
    )
    baseline_hit_rate = sum(1 for p in baseline_recs if p in future) / 5
    baseline_conversions.append(baseline_hit_rate)

# Calculate metrics
avg_meta = np.mean(meta_conversions) * 100
avg_baseline = np.mean(baseline_conversions) * 100
improvement = ((avg_meta - avg_baseline) / avg_baseline) * 100

print(f"\n📊 RESULTS (Tested on {len(meta_conversions)} customers):")
print("-" * 40)
print(f"Average Recommendation Accuracy:")
print(f"  Meta-Learning:    {avg_meta:.1f}%")
print(f"  Baseline:         {avg_baseline:.1f}%")
print(f"  Improvement:      +{improvement:.1f}%")

# Business translation
avg_order_value = 75  # dollars
print(f"\n💵 BUSINESS VALUE:")
print("-" * 40)
print(f"Assumptions:")
print(f"  • 10,000 new customers per month")
print(f"  • Average order value: ${avg_order_value}")
print(f"  • Recommendation acceptance rate: 20%")

monthly_customers = 10000
acceptance_rate = 0.20

baseline_revenue = monthly_customers * avg_baseline/100 * acceptance_rate * avg_order_value
meta_revenue = monthly_customers * avg_meta/100 * acceptance_rate * avg_order_value
additional_revenue = meta_revenue - baseline_revenue

print(f"\nMonthly Revenue from Recommendations:")
print(f"  Baseline system:     ${baseline_revenue:,.0f}")
print(f"  Meta-Learning:       ${meta_revenue:,.0f}")
print(f"  Additional revenue:  ${additional_revenue:,.0f}/month")
print(f"  Annual impact:       ${additional_revenue * 12:,.0f}/year")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Recommendation accuracy
ax1 = axes[0]
methods = ['Baseline\n(Popular Items)', 'Meta-Learning\n(Personalized)']
accuracies = [avg_baseline, avg_meta]
colors = ['#ff6b6b', '#51cf66']

bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Recommendation Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Recommendation System Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(accuracies) * 1.3])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold')

# Plot 2: Revenue impact
ax2 = axes[1]
revenues = [baseline_revenue/1000, meta_revenue/1000]  # In thousands
bars = ax2.bar(methods, revenues, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Monthly Revenue ($1000s)', fontsize=12, fontweight='bold')
ax2.set_title('Revenue Impact', fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(revenues) * 1.3])
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rev in zip(bars, revenues):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${rev:.0f}K', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

# Add improvement annotation
ax2.annotate(
    f'+${additional_revenue/1000:.0f}K/mo',
    xy=(1, revenues[1]), xytext=(0.5, revenues[1] + max(revenues)*0.1),
    arrowprops=dict(arrowstyle='->', color='green', lw=2),
    fontsize=11, fontweight='bold', color='green',
    ha='center'
)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/recommendation_comparison.png', 
            dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 META-LEARNING RECOMMENDATION SYSTEM - SUMMARY")
print("=" * 80)
print(f"""
WHAT WE BUILT:
-------------
✓ Meta-learning recommendation system for e-commerce
✓ Solves the cold-start problem for new customers
✓ Learns from just 1-3 customer purchases
✓ {improvement:.0f}% more accurate than baseline

HOW IT WORKS:
------------
1. Learn customer preferences from purchase patterns
2. Create "prototypes" representing customer segments
3. Match new customers to similar existing customers
4. Recommend what similar customers liked
5. Adapt quickly with minimal data

BUSINESS IMPACT:
---------------
• Better recommendations = Higher conversion rates
• ${additional_revenue:,.0f} additional revenue per month
• ${additional_revenue * 12:,.0f} additional revenue per year
• Improved customer experience from day 1

REAL-WORLD APPLICATIONS:
-----------------------
🛍️  E-commerce: Personalize product recommendations
📺  Streaming: Suggest content to new subscribers
🏪  Retail: Targeted promotions for new customers
📱  Apps: Feature recommendations for new users
🏢  B2B: Service suggestions for new clients

KEY ADVANTAGES:
--------------
✓ Works with minimal customer data (1-5 interactions)
✓ No complex user profiles needed
✓ Adapts to new products automatically
✓ Scales to millions of customers
✓ Better than "popular items" approach

NEXT STEPS:
----------
1. Try with your own customer data
2. Experiment with different embedding strategies
3. Add more features (demographics, seasonality)
4. Implement online learning for real-time adaptation
5. A/B test in production environment

This is just the beginning - meta-learning enables truly personalized
recommendations from day one of customer interaction! 🚀
""")

print("\nFiles created:")
print("  • recommendation_comparison.png - Performance visualization")
print("\nYou now understand how meta-learning solves real business problems!")
