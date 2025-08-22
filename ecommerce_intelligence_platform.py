# Real-Time E-commerce Customer Intelligence Platform
# Complete Python Implementation for Portfolio Project
# Author: Data Analyst Portfolio Project
# Date: August 22, 2025

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import sqlite3
import streamlit as st


class EcommerceCustomerIntelligence:
    """
    Real-Time E-commerce Customer Intelligence Platform
    
    This class implements a comprehensive analytics platform for e-commerce
    customer intelligence, including RFM analysis, CLV prediction, churn analysis,
    and real-time dashboard capabilities.
    """
    
    def __init__(self):
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.sessions_df = None
        self.rfm_df = None
        self.clv_model = None
        self.churn_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all e-commerce datasets"""
        try:
            self.products_df = pd.read_csv('ecommerce_products.csv')
            self.customers_df = pd.read_csv('ecommerce_customers.csv')
            self.transactions_df = pd.read_csv('ecommerce_transactions.csv')
            self.sessions_df = pd.read_csv('ecommerce_sessions.csv')
            
            # Convert date columns
            self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
            self.customers_df['last_purchase_date'] = pd.to_datetime(self.customers_df['last_purchase_date'])
            self.transactions_df['order_date'] = pd.to_datetime(self.transactions_df['order_date'])
            self.sessions_df['event_time'] = pd.to_datetime(self.sessions_df['event_time'])
            
            print("✅ Data loaded successfully!")
            print(f"Products: {len(self.products_df)} | Customers: {len(self.customers_df)}")
            print(f"Transactions: {len(self.transactions_df)} | Session Events: {len(self.sessions_df)}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def perform_eda(self):
        """Perform comprehensive Exploratory Data Analysis"""
        
        print("🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # 1. Business Overview Metrics
        total_revenue = self.transactions_df['final_total'].sum()
        total_orders = len(self.transactions_df)
        avg_order_value = self.transactions_df['final_total'].mean()
        total_customers = len(self.customers_df)
        active_customers = self.customers_df[self.customers_df['total_orders'] > 0].shape[0]
        
        print(f"📈 BUSINESS OVERVIEW")
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Total Orders: {total_orders:,}")
        print(f"Average Order Value: ${avg_order_value:.2f}")
        print(f"Total Customers: {total_customers:,}")
        print(f"Active Customers: {active_customers:,} ({active_customers/total_customers*100:.1f}%)")
        
        # 2. Product Performance Analysis
        print(f"\n📦 PRODUCT PERFORMANCE")
        
        # Calculate product metrics from transactions
        product_performance = []
        for _, product in self.products_df.iterrows():
            product_transactions = []
            
            # Find all transactions containing this product
            for _, transaction in self.transactions_df.iterrows():
                items = eval(transaction['items']) if isinstance(transaction['items'], str) else transaction['items']
                for item in items:
                    if item['product_id'] == product['product_id']:
                        product_transactions.append({
                            'quantity': item['quantity'],
                            'revenue': item['item_total'],
                            'date': transaction['order_date']
                        })
            
            if product_transactions:
                total_quantity = sum([t['quantity'] for t in product_transactions])
                total_revenue = sum([t['revenue'] for t in product_transactions])
                
                product_performance.append({
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': product['price'],
                    'total_sold': total_quantity,
                    'total_revenue': total_revenue,
                    'num_orders': len(product_transactions)
                })
        
        product_perf_df = pd.DataFrame(product_performance)
        
        if not product_perf_df.empty:
            top_products = product_perf_df.nlargest(5, 'total_revenue')
            print(f"Top 5 Products by Revenue:")
            for _, product in top_products.iterrows():
                print(f"  {product['product_name']}: ${product['total_revenue']:.2f}")
        
        # 3. Customer Segmentation Overview
        print(f"\n👥 CUSTOMER SEGMENTATION")
        segment_analysis = self.customers_df['customer_segment'].value_counts()
        for segment, count in segment_analysis.items():
            avg_spent = self.customers_df[self.customers_df['customer_segment'] == segment]['total_spent'].mean()
            print(f"{segment}: {count} customers (${avg_spent:.2f} avg spent)")
        
        # 4. Seasonal Trends
        print(f"\n📅 SEASONAL TRENDS")
        monthly_sales = self.transactions_df.groupby(self.transactions_df['order_date'].dt.to_period('M'))['final_total'].agg(['sum', 'count'])
        print("Monthly Sales (Last 6 months):")
        for month, data in monthly_sales.tail(6).iterrows():
            print(f"  {month}: ${data['sum']:,.0f} ({data['count']} orders)")
        
        return {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'active_customers': active_customers,
            'product_performance': product_perf_df,
            'monthly_sales': monthly_sales
        }
    
    def calculate_rfm_analysis(self):
        """Calculate RFM (Recency, Frequency, Monetary) Analysis"""
        
        print("\n🎯 RFM ANALYSIS")
        print("=" * 30)
        
        # Calculate RFM metrics for each customer
        rfm_data = []
        analysis_date = datetime(2025, 8, 22)  # Current date for analysis
        
        for _, customer in self.customers_df.iterrows():
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer['customer_id']
            ]
            
            if len(customer_transactions) > 0:
                # Recency: Days since last purchase
                last_purchase = customer_transactions['order_date'].max()
                recency = (analysis_date - last_purchase).days
                
                # Frequency: Number of purchases
                frequency = len(customer_transactions)
                
                # Monetary: Total amount spent
                monetary = customer_transactions['final_total'].sum()
                
                rfm_data.append({
                    'customer_id': customer['customer_id'],
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary,
                    'age': customer['age'],
                    'gender': customer['gender'],
                    'location': customer['location'],
                    'customer_segment': customer['customer_segment']
                })
        
        self.rfm_df = pd.DataFrame(rfm_data)
        
        if self.rfm_df.empty:
            print("❌ No RFM data available")
            return None
        
        # Calculate RFM scores (1-5 scale)
        self.rfm_df['R_score'] = pd.qcut(self.rfm_df['recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
        self.rfm_df['F_score'] = pd.qcut(self.rfm_df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        self.rfm_df['M_score'] = pd.qcut(self.rfm_df['monetary'], 5, labels=[1,2,3,4,5])
        
        # Create RFM segments
        self.rfm_df['RFM_score'] = self.rfm_df['R_score'].astype(str) + self.rfm_df['F_score'].astype(str) + self.rfm_df['M_score'].astype(str)
        
        # Define customer segments based on RFM scores
        def segment_customers(row):
            if row['RFM_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'New Customers'
            elif row['RFM_score'] in ['155', '251', '252', '231', '241', '151']:
                return 'Promising'
            elif row['RFM_score'] in ['155', '132', '123', '122', '212', '211']:
                return 'Need Attention'
            elif row['RFM_score'] in ['155', '144', '214', '215', '115', '114']:
                return 'About to Sleep'
            elif row['RFM_score'] in ['155', '133', '134', '143', '244', '334']:
                return 'At Risk'
            elif row['RFM_score'] in ['155', '111', '112', '121', '131', '141']:
                return 'Cannot Lose Them'
            else:
                return 'Hibernating'
        
        self.rfm_df['segment'] = self.rfm_df.apply(segment_customers, axis=1)
        
        # RFM Summary
        print(f"RFM Analysis Summary:")
        segment_summary = self.rfm_df['segment'].value_counts()
        for segment, count in segment_summary.items():
            avg_monetary = self.rfm_df[self.rfm_df['segment'] == segment]['monetary'].mean()
            print(f"  {segment}: {count} customers (${avg_monetary:.2f} avg value)")
        
        return self.rfm_df
    
    def predict_customer_lifetime_value(self):
        """Build Customer Lifetime Value (CLV) prediction model"""
        
        print("\n💰 CUSTOMER LIFETIME VALUE PREDICTION")
        print("=" * 45)
        
        if self.rfm_df is None:
            print("❌ RFM analysis required first")
            return None
        
        # Prepare features for CLV prediction
        features = self.rfm_df.copy()
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_location = LabelEncoder()
        le_segment = LabelEncoder()
        
        features['gender_encoded'] = le_gender.fit_transform(features['gender'])
        features['location_encoded'] = le_location.fit_transform(features['location'])
        features['segment_encoded'] = le_segment.fit_transform(features['customer_segment'])
        
        # Calculate additional features
        features['avg_order_value'] = features['monetary'] / features['frequency']
        features['days_per_order'] = features['recency'] / features['frequency']
        
        # Select features for prediction
        feature_columns = [
            'recency', 'frequency', 'age', 'gender_encoded', 
            'location_encoded', 'segment_encoded', 'avg_order_value', 'days_per_order'
        ]
        
        X = features[feature_columns].fillna(0)
        y = features['monetary']  # Predict total monetary value
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model for CLV prediction
        self.clv_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clv_model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = self.clv_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"CLV Model Performance:")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        print(f"  R² Score: {r2:.3f}")
        
        # Predict CLV for all customers
        clv_predictions = self.clv_model.predict(X)
        features['predicted_clv'] = clv_predictions
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.clv_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop CLV Prediction Features:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return features
    
    def predict_customer_churn(self):
        """Build Customer Churn Prediction model"""
        
        print("\n⚠️  CUSTOMER CHURN PREDICTION")
        print("=" * 35)
        
        if self.rfm_df is None:
            print("❌ RFM analysis required first")
            return None
        
        # Define churn (customers who haven't purchased in last 90 days)
        churn_threshold = 90
        self.rfm_df['is_churned'] = (self.rfm_df['recency'] > churn_threshold).astype(int)
        
        # Prepare features
        features = self.rfm_df.copy()
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_location = LabelEncoder()
        le_segment = LabelEncoder()
        
        features['gender_encoded'] = le_gender.fit_transform(features['gender'])
        features['location_encoded'] = le_location.fit_transform(features['location'])
        features['segment_encoded'] = le_segment.fit_transform(features['customer_segment'])
        
        # Additional features
        features['avg_order_value'] = features['monetary'] / features['frequency']
        features['purchase_frequency'] = features['frequency'] / (features['recency'] + 1)
        
        # Select features
        feature_columns = [
            'recency', 'frequency', 'monetary', 'age', 
            'gender_encoded', 'location_encoded', 'segment_encoded',
            'avg_order_value', 'purchase_frequency'
        ]
        
        X = features[feature_columns].fillna(0)
        y = features['is_churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Gradient Boosting model for churn prediction
        self.churn_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.churn_model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = self.churn_model.predict(X_test)
        y_pred_proba = self.churn_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Churn Model Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Churn Rate: {y.mean():.1%}")
        
        # Predict churn probability for all customers
        churn_probabilities = self.churn_model.predict_proba(X)[:, 1]
        features['churn_probability'] = churn_probabilities
        
        # Classify churn risk
        features['churn_risk'] = pd.cut(
            features['churn_probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.churn_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Churn Prediction Features:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        churn_risk_summary = features['churn_risk'].value_counts()
        print(f"\nChurn Risk Distribution:")
        for risk, count in churn_risk_summary.items():
            print(f"  {risk} Risk: {count} customers ({count/len(features)*100:.1f}%)")
        
        return features
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        
        print("\n📊 CREATING VISUALIZATIONS")
        print("=" * 35)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('E-commerce Customer Intelligence Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Revenue Trends Over Time
        monthly_revenue = self.transactions_df.groupby(
            self.transactions_df['order_date'].dt.to_period('M')
        )['final_total'].sum()
        
        monthly_revenue.plot(kind='line', ax=axes[0, 0], marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Monthly Revenue Trends', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Customer Segments Distribution
        if self.rfm_df is not None:
            segment_counts = self.rfm_df['segment'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            wedges, texts, autotexts = axes[0, 1].pie(
                segment_counts.values, 
                labels=segment_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            axes[0, 1].set_title('RFM Customer Segments', fontsize=14, fontweight='bold')
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'RFM Analysis\nNot Available', ha='center', va='center')
            axes[0, 1].set_title('RFM Customer Segments', fontsize=14, fontweight='bold')
        
        # 3. Product Category Performance
        category_performance = []
        for category in self.products_df['category'].unique():
            category_products = self.products_df[self.products_df['category'] == category]['product_id']
            
            category_revenue = 0
            for _, transaction in self.transactions_df.iterrows():
                items = eval(transaction['items']) if isinstance(transaction['items'], str) else transaction['items']
                for item in items:
                    if item['product_id'] in category_products.values:
                        category_revenue += item['item_total']
            
            category_performance.append({
                'category': category,
                'revenue': category_revenue
            })
        
        category_df = pd.DataFrame(category_performance).sort_values('revenue', ascending=True)
        
        bars = axes[1, 0].barh(category_df['category'], category_df['revenue'], color='skyblue')
        axes[1, 0].set_title('Revenue by Product Category', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Revenue ($)')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            axes[1, 0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                           f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        # 4. Customer Age Distribution
        ages = self.customers_df['age'].dropna()
        axes[1, 1].hist(ages, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Customer Age Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_age = ages.mean()
        axes[1, 1].axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean Age: {mean_age:.1f}')
        axes[1, 1].legend()
        
        # 5. Order Value Distribution
        order_values = self.transactions_df['final_total']
        axes[2, 0].hist(order_values, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[2, 0].set_title('Order Value Distribution', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Order Value ($)')
        axes[2, 0].set_ylabel('Number of Orders')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_order = order_values.mean()
        median_order = order_values.median()
        axes[2, 0].axvline(mean_order, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: ${mean_order:.2f}')
        axes[2, 0].axvline(median_order, color='blue', linestyle='--', linewidth=2, 
                          label=f'Median: ${median_order:.2f}')
        axes[2, 0].legend()
        
        # 6. Customer Churn Risk (if available)
        if self.rfm_df is not None and 'churn_risk' in self.rfm_df.columns:
            churn_counts = self.rfm_df['churn_risk'].value_counts()
            colors = ['green', 'yellow', 'red']
            bars = axes[2, 1].bar(churn_counts.index, churn_counts.values, color=colors)
            axes[2, 1].set_title('Customer Churn Risk Distribution', fontsize=14, fontweight='bold')
            axes[2, 1].set_ylabel('Number of Customers')
            
            # Add value labels on bars
            for bar, value in zip(bars, churn_counts.values):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[2, 1].text(0.5, 0.5, 'Churn Analysis\nNot Available', ha='center', va='center')
            axes[2, 1].set_title('Customer Churn Risk Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ecommerce_intelligence_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive dashboard saved as 'ecommerce_intelligence_dashboard.png'")
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        
        print("\n💡 KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 55)
        
        # 1. Revenue Insights
        total_revenue = self.transactions_df['final_total'].sum()
        avg_order_value = self.transactions_df['final_total'].mean()
        
        print(f"📈 REVENUE OPTIMIZATION:")
        print(f"   • Current AOV: ${avg_order_value:.2f}")
        if avg_order_value < 75:
            print(f"   • RECOMMENDATION: Implement upselling strategies to increase AOV to $75+")
        
        # 2. Customer Segment Insights
        if self.rfm_df is not None:
            champions = len(self.rfm_df[self.rfm_df['segment'] == 'Champions'])
            at_risk = len(self.rfm_df[self.rfm_df['segment'] == 'At Risk'])
            
            print(f"\n👥 CUSTOMER RETENTION:")
            print(f"   • Champions: {champions} customers")
            print(f"   • At Risk: {at_risk} customers")
            
            if at_risk > champions * 0.5:
                print(f"   • URGENT: {at_risk} customers at risk - implement retention campaign")
            
            # High-value customer insights
            high_value_customers = self.rfm_df[self.rfm_df['monetary'] > self.rfm_df['monetary'].quantile(0.8)]
            print(f"   • Top 20% customers generate {high_value_customers['monetary'].sum()/self.rfm_df['monetary'].sum()*100:.1f}% of revenue")
        
        # 3. Product Performance Insights
        category_revenue = {}
        for category in self.products_df['category'].unique():
            category_products = self.products_df[self.products_df['category'] == category]['product_id']
            revenue = 0
            for _, transaction in self.transactions_df.iterrows():
                items = eval(transaction['items']) if isinstance(transaction['items'], str) else transaction['items']
                for item in items:
                    if item['product_id'] in category_products.values:
                        revenue += item['item_total']
            category_revenue[category] = revenue
        
        top_category = max(category_revenue, key=category_revenue.get)
        worst_category = min(category_revenue, key=category_revenue.get)
        
        print(f"\n📦 PRODUCT STRATEGY:")
        print(f"   • Top Category: {top_category} (${category_revenue[top_category]:,.0f})")
        print(f"   • Underperforming: {worst_category} (${category_revenue[worst_category]:,.0f})")
        print(f"   • RECOMMENDATION: Expand {top_category} inventory, optimize {worst_category} pricing")
        
        # 4. Seasonal Insights
        monthly_orders = self.transactions_df.groupby(
            self.transactions_df['order_date'].dt.month
        )['transaction_id'].count()
        
        peak_month = monthly_orders.idxmax()
        low_month = monthly_orders.idxmin()
        
        print(f"\n📅 SEASONAL PATTERNS:")
        print(f"   • Peak Month: {peak_month} ({monthly_orders[peak_month]} orders)")
        print(f"   • Low Month: {low_month} ({monthly_orders[low_month]} orders)")
        print(f"   • RECOMMENDATION: Plan inventory and marketing campaigns around seasonal peaks")
        
        # 5. Churn Prevention
        if self.rfm_df is not None and 'churn_probability' in self.rfm_df.columns:
            high_churn_risk = len(self.rfm_df[self.rfm_df['churn_probability'] > 0.7])
            
            print(f"\n⚠️  CHURN PREVENTION:")
            print(f"   • High Churn Risk: {high_churn_risk} customers")
            if high_churn_risk > 0:
                avg_value_at_risk = self.rfm_df[self.rfm_df['churn_probability'] > 0.7]['monetary'].mean()
                potential_loss = high_churn_risk * avg_value_at_risk
                print(f"   • Potential Revenue Loss: ${potential_loss:,.2f}")
                print(f"   • URGENT: Launch retention campaign for high-risk customers")
        
        # 6. Growth Opportunities
        print(f"\n🚀 GROWTH OPPORTUNITIES:")
        
        # New vs returning customers
        returning_customers = len(self.customers_df[self.customers_df['total_orders'] > 1])
        new_customers = len(self.customers_df) - returning_customers
        
        print(f"   • New Customers: {new_customers}")
        print(f"   • Returning Customers: {returning_customers}")
        print(f"   • Repeat Rate: {returning_customers/len(self.customers_df)*100:.1f}%")
        
        if returning_customers/len(self.customers_df) < 0.3:
            print(f"   • RECOMMENDATION: Focus on customer retention programs")
        
        # Geographic expansion opportunities
        location_revenue = self.customers_df.groupby('location')['total_spent'].sum().sort_values(ascending=False)
        print(f"   • Top Market: {location_revenue.index[0]} (${location_revenue.iloc[0]:,.0f})")
        print(f"   • RECOMMENDATION: Expand marketing in underperforming regions")
    
    def export_results(self):
        """Export analysis results and models"""
        
        print("\n💾 EXPORTING RESULTS")
        print("=" * 25)
        
        # Save RFM analysis
        if self.rfm_df is not None:
            self.rfm_df.to_csv('customer_rfm_analysis.csv', index=False)
            print("✅ RFM analysis exported to 'customer_rfm_analysis.csv'")
        
        # Save customer insights summary
        customer_insights = {
            'total_customers': len(self.customers_df),
            'active_customers': len(self.customers_df[self.customers_df['total_orders'] > 0]),
            'total_revenue': float(self.transactions_df['final_total'].sum()),
            'avg_order_value': float(self.transactions_df['final_total'].mean()),
            'total_orders': len(self.transactions_df),
            'analysis_date': datetime.now().isoformat()
        }
        
        # Add RFM segment summary
        if self.rfm_df is not None:
            segment_summary = self.rfm_df.groupby('segment').agg({
                'customer_id': 'count',
                'monetary': 'mean',
                'frequency': 'mean',
                'recency': 'mean'
            }).round(2)
            
            customer_insights['rfm_segments'] = segment_summary.to_dict('index')
        
        # Add churn analysis
        if self.rfm_df is not None and 'churn_probability' in self.rfm_df.columns:
            churn_summary = {
                'high_risk_customers': len(self.rfm_df[self.rfm_df['churn_probability'] > 0.7]),
                'medium_risk_customers': len(self.rfm_df[(self.rfm_df['churn_probability'] > 0.3) & (self.rfm_df['churn_probability'] <= 0.7)]),
                'low_risk_customers': len(self.rfm_df[self.rfm_df['churn_probability'] <= 0.3])
            }
            customer_insights['churn_analysis'] = churn_summary
        
        # Save to JSON
        with open('customer_intelligence_summary.json', 'w') as f:
            json.dump(customer_insights, f, indent=2, default=str)
        
        print("✅ Customer insights exported to 'customer_intelligence_summary.json'")
        
        # Create executive summary
        executive_summary = f"""
# EXECUTIVE SUMMARY - E-COMMERCE CUSTOMER INTELLIGENCE

## Key Performance Indicators
- Total Revenue: ${self.transactions_df['final_total'].sum():,.2f}
- Total Customers: {len(self.customers_df):,}
- Active Customers: {len(self.customers_df[self.customers_df['total_orders'] > 0]):,}
- Average Order Value: ${self.transactions_df['final_total'].mean():.2f}
- Total Orders: {len(self.transactions_df):,}

## Strategic Recommendations

### 1. Revenue Optimization
- Implement cross-selling and upselling strategies
- Focus on increasing average order value through bundling
- Optimize pricing for underperforming categories

### 2. Customer Retention
- Launch targeted campaigns for at-risk customers
- Develop loyalty programs for Champions and Loyal Customers
- Create personalized experiences for high-value segments

### 3. Growth Opportunities
- Expand successful product categories
- Target underperforming geographic markets
- Improve new customer acquisition and retention rates

### 4. Operational Efficiency
- Optimize inventory based on seasonal demand patterns
- Automate churn prediction and intervention workflows
- Implement real-time customer scoring for personalization

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('executive_summary.md', 'w') as f:
            f.write(executive_summary)
        
        print("✅ Executive summary exported to 'executive_summary.md'")
        print("✅ Dashboard visualization saved as 'ecommerce_intelligence_dashboard.png'")
        
        return customer_insights
    
    def run_complete_analysis(self):
        """Run the complete customer intelligence analysis"""
        
        print("🚀 E-COMMERCE CUSTOMER INTELLIGENCE PLATFORM")
        print("=" * 60)
        print("Starting comprehensive customer analytics...")
        
        # Step 1: Load Data
        if not self.load_data():
            return None
        
        # Step 2: Exploratory Data Analysis
        eda_results = self.perform_eda()
        
        # Step 3: RFM Analysis
        rfm_results = self.calculate_rfm_analysis()
        
        # Step 4: Customer Lifetime Value Prediction
        clv_results = self.predict_customer_lifetime_value()
        
        # Step 5: Churn Prediction
        churn_results = self.predict_customer_churn()
        
        # Step 6: Create Visualizations
        self.create_visualizations()
        
        # Step 7: Generate Business Insights
        self.generate_business_insights()
        
        # Step 8: Export Results
        results = self.export_results()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print("=" * 30)
        print("📁 Files Generated:")
        print("   • customer_rfm_analysis.csv")
        print("   • customer_intelligence_summary.json")
        print("   • executive_summary.md")
        print("   • ecommerce_intelligence_dashboard.png")
        
        return results


def main():
    """Main function to run the complete e-commerce intelligence analysis"""
    
    # Initialize the platform
    platform = EcommerceCustomerIntelligence()
    
    # Run complete analysis
    results = platform.run_complete_analysis()
    
    if results:
        print(f"\n🎉 SUCCESS!")
        print("Your E-commerce Customer Intelligence Platform is ready for presentation!")
        print("\n📊 Key Metrics:")
        print(f"   Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"   Active Customers: {results['active_customers']:,}")
        print(f"   Average Order Value: ${results['avg_order_value']:.2f}")
        
        print(f"\n💼 Portfolio Impact:")
        print("   ✅ Demonstrates advanced analytics skills")
        print("   ✅ Shows real-world business problem solving")
        print("   ✅ Includes machine learning and predictive modeling")
        print("   ✅ Provides actionable business insights")
        print("   ✅ Professional data visualization and reporting")
    
    return results


if __name__ == "__main__":
    main()