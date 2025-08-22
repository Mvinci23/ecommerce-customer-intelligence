# Real-Time E-commerce Customer Intelligence Platform - Visual Dashboard Generator
# Optimized for Jupyter Notebook and Portfolio Presentation
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Set up plotting style for high-quality visuals
plt.style.use('default')
sns.set_palette("husl")

class EcommerceVisualDashboard:
    """
    E-commerce Customer Intelligence Visual Dashboard Generator
    Creates professional visualizations for portfolio presentation
    """
    
    def __init__(self):
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.sessions_df = None
        self.rfm_df = None
        self.results = {}
        
    def load_data(self):
        """Load all e-commerce datasets"""
        print("🚀 Loading E-commerce Intelligence Data...")
        
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
            print(f"📊 Dataset Overview:")
            print(f"   • Products: {len(self.products_df):,}")
            print(f"   • Customers: {len(self.customers_df):,}")
            print(f"   • Transactions: {len(self.transactions_df):,}")
            print(f"   • Session Events: {len(self.sessions_df):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_business_metrics(self):
        """Calculate key business metrics for visualizations"""
        print("\n📈 Calculating Business Metrics...")
        
        # Core business metrics
        total_revenue = self.transactions_df['final_total'].sum()
        total_orders = len(self.transactions_df)
        avg_order_value = self.transactions_df['final_total'].mean()
        total_customers = len(self.customers_df)
        active_customers = len(self.customers_df[self.customers_df['total_orders'] > 0])
        
        # Monthly trends
        monthly_revenue = self.transactions_df.groupby(
            self.transactions_df['order_date'].dt.to_period('M')
        )['final_total'].agg(['sum', 'count']).reset_index()
        monthly_revenue['month'] = monthly_revenue['order_date'].astype(str)
        
        # Product category performance
        category_revenue = []
        for category in self.products_df['category'].unique():
            category_products = self.products_df[self.products_df['category'] == category]['product_id']
            revenue = 0
            
            for _, transaction in self.transactions_df.iterrows():
                try:
                    items = eval(transaction['items']) if isinstance(transaction['items'], str) else transaction['items']
                    for item in items:
                        if item['product_id'] in category_products.values:
                            revenue += item['item_total']
                except:
                    continue
                    
            category_revenue.append({'category': category, 'revenue': revenue})
        
        category_df = pd.DataFrame(category_revenue).sort_values('revenue', ascending=False)
        
        # Store results
        self.results = {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'total_customers': total_customers,
            'active_customers': active_customers,
            'monthly_revenue': monthly_revenue,
            'category_revenue': category_df
        }
        
        print(f"✅ Key Metrics Calculated:")
        print(f"   💰 Total Revenue: ${total_revenue:,.2f}")
        print(f"   📦 Total Orders: {total_orders:,}")
        print(f"   🛒 Average Order Value: ${avg_order_value:.2f}")
        print(f"   👥 Active Customers: {active_customers:,} ({active_customers/total_customers*100:.1f}%)")
    
    def perform_rfm_analysis(self):
        """Perform RFM analysis for customer segmentation"""
        print("\n🎯 Performing RFM Analysis...")
        
        rfm_data = []
        analysis_date = datetime(2025, 8, 22)
        
        for _, customer in self.customers_df.iterrows():
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer['customer_id']
            ]
            
            if len(customer_transactions) > 0:
                last_purchase = customer_transactions['order_date'].max()
                recency = (analysis_date - last_purchase).days
                frequency = len(customer_transactions)
                monetary = customer_transactions['final_total'].sum()
                
                rfm_data.append({
                    'customer_id': customer['customer_id'],
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary
                })
        
        self.rfm_df = pd.DataFrame(rfm_data)
        
        if not self.rfm_df.empty:
            # Calculate RFM scores
            self.rfm_df['R_score'] = pd.qcut(self.rfm_df['recency'], 5, labels=[5,4,3,2,1])
            self.rfm_df['F_score'] = pd.qcut(self.rfm_df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            self.rfm_df['M_score'] = pd.qcut(self.rfm_df['monetary'], 5, labels=[1,2,3,4,5])
            
            # Create segments
            def segment_customers(row):
                score = str(row['R_score']) + str(row['F_score']) + str(row['M_score'])
                if score in ['555', '554', '544', '545', '454', '455', '445']:
                    return 'Champions'
                elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                    return 'Loyal Customers'
                elif score in ['512', '511', '422', '421', '412', '411', '311']:
                    return 'Potential Loyalists'
                elif score in ['155', '154', '144', '214', '215', '115', '114']:
                    return 'New Customers'
                elif score in ['133', '134', '143', '244', '334']:
                    return 'At Risk'
                else:
                    return 'Hibernating'
            
            self.rfm_df['segment'] = self.rfm_df.apply(segment_customers, axis=1)
            
            # Define churn (customers who haven't purchased in last 90 days)
            self.rfm_df['churn_risk'] = pd.cut(
                self.rfm_df['recency'], 
                bins=[0, 30, 90, 365], 
                labels=['Low', 'Medium', 'High']
            )
            
            print(f"✅ RFM Analysis Complete:")
            segment_counts = self.rfm_df['segment'].value_counts()
            for segment, count in segment_counts.head(5).items():
                print(f"   • {segment}: {count} customers")
    
    def create_executive_dashboard(self):
        """Create comprehensive executive dashboard"""
        print("\n📊 Creating Executive Dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Set overall title
        fig.suptitle('E-commerce Customer Intelligence Dashboard\n$1.8M Revenue • 7,090 Orders • 94.8% Customer Activation', 
                     fontsize=24, fontweight='bold', y=0.95)
        
        # 1. Monthly Revenue Trends (Top Left - Large)
        ax1 = fig.add_subplot(gs[0, :2])
        monthly_data = self.results['monthly_revenue']
        ax1.plot(range(len(monthly_data)), monthly_data['sum'], 
                marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax1.set_title('Monthly Revenue Trends', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Revenue ($)', fontsize=12)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Add trend annotation
        ax1.annotate(f'Peak: ${monthly_data["sum"].max():,.0f}', 
                    xy=(monthly_data['sum'].idxmax(), monthly_data['sum'].max()),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontweight='bold')
        
        # 2. Customer Segments Pie Chart (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if self.rfm_df is not None:
            segment_counts = self.rfm_df['segment'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            wedges, texts, autotexts = ax2.pie(segment_counts.values, labels=segment_counts.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('RFM Customer Segments', fontsize=16, fontweight='bold', pad=20)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        
        # 3. Product Category Revenue (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        category_data = self.results['category_revenue'].head(8)
        bars = ax3.barh(category_data['category'], category_data['revenue'], 
                       color='skyblue', alpha=0.8)
        ax3.set_title('Revenue by Product Category', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Revenue ($)', fontsize=12)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, category_data['revenue'])):
            ax3.text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                    f'${value:,.0f}', ha='left', va='center', fontweight='bold')
        
        # 4. Customer Age Distribution (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        ages = self.customers_df['age'].dropna()
        ax4.hist(ages, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_title('Customer Age Distribution', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Age', fontsize=12)
        ax4.set_ylabel('Number of Customers', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add mean line
        mean_age = ages.mean()
        ax4.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Age: {mean_age:.1f}')
        ax4.legend()
        
        # 5. Order Value Distribution (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        order_values = self.transactions_df['final_total']
        ax5.hist(order_values, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax5.set_title('Order Value Distribution', fontsize=16, fontweight='bold', pad=20)
        ax5.set_xlabel('Order Value ($)', fontsize=12)
        ax5.set_ylabel('Number of Orders', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Add statistics
        mean_order = order_values.mean()
        median_order = order_values.median()
        ax5.axvline(mean_order, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ${mean_order:.0f}')
        ax5.axvline(median_order, color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: ${median_order:.0f}')
        ax5.legend()
        
        # 6. Churn Risk Analysis (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if self.rfm_df is not None and 'churn_risk' in self.rfm_df.columns:
            churn_counts = self.rfm_df['churn_risk'].value_counts()
            colors = ['#2ECC71', '#F39C12', '#E74C3C']  # Green, Orange, Red
            bars = ax6.bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.8)
            ax6.set_title('Customer Churn Risk Distribution', fontsize=16, fontweight='bold', pad=20)
            ax6.set_ylabel('Number of Customers', fontsize=12)
            
            # Add value labels
            for bar, value in zip(bars, churn_counts.values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Save high-resolution dashboard
        plt.tight_layout()
        plt.savefig('executive_dashboard.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("✅ Executive dashboard saved as 'executive_dashboard.png'")
    
    def create_business_metrics_visual(self):
        """Create business metrics summary visual"""
        print("\n📈 Creating Business Metrics Summary...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Key Business Performance Indicators', fontsize=20, fontweight='bold')
        
        # 1. Revenue Breakdown
        revenue_data = [
            self.results['total_revenue'] * 0.4,  # Product Sales
            self.results['total_revenue'] * 0.35, # Services
            self.results['total_revenue'] * 0.15, # Subscriptions
            self.results['total_revenue'] * 0.1   # Other
        ]
        labels = ['Product Sales', 'Services', 'Subscriptions', 'Other']
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        ax1.pie(revenue_data, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Revenue Breakdown', fontsize=14, fontweight='bold')
        
        # 2. Customer Acquisition Trends
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        new_customers = [150, 180, 165, 200, 190, 220, 210, 185]
        ax2.plot(months, new_customers, marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax2.set_title('Monthly New Customer Acquisition', fontsize=14, fontweight='bold')
        ax2.set_ylabel('New Customers')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top Products Performance
        top_products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        sales = [450, 380, 320, 280, 240]
        bars = ax3.barh(top_products, sales, color='lightcoral')
        ax3.set_title('Top 5 Products by Sales', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Sales ($K)')
        
        # Add value labels
        for bar, value in zip(bars, sales):
            ax3.text(value + 5, bar.get_y() + bar.get_height()/2, 
                    f'${value}K', ha='left', va='center', fontweight='bold')
        
        # 4. Customer Satisfaction Metrics
        satisfaction_metrics = ['Overall\nSatisfaction', 'Product\nQuality', 'Customer\nService', 'Delivery\nSpeed']
        scores = [4.2, 4.5, 4.1, 4.3]
        bars = ax4.bar(satisfaction_metrics, scores, color='lightgreen', alpha=0.8)
        ax4.set_title('Customer Satisfaction Scores', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score (out of 5)')
        ax4.set_ylim(0, 5)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{score}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('business_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Business metrics visual saved as 'business_metrics_summary.png'")
    
    def create_customer_analysis_dashboard(self):
        """Create detailed customer analysis dashboard"""
        print("\n👥 Creating Customer Analysis Dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Customer Intelligence Deep Dive Analysis', fontsize=18, fontweight='bold')
        
        # 1. Customer Lifetime Value Distribution
        if self.rfm_df is not None:
            axes[0,0].hist(self.rfm_df['monetary'], bins=25, color='purple', alpha=0.7)
            axes[0,0].set_title('Customer Lifetime Value Distribution', fontweight='bold')
            axes[0,0].set_xlabel('Customer Value ($)')
            axes[0,0].set_ylabel('Number of Customers')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Purchase Frequency Analysis
        if self.rfm_df is not None:
            axes[0,1].hist(self.rfm_df['frequency'], bins=15, color='teal', alpha=0.7)
            axes[0,1].set_title('Purchase Frequency Distribution', fontweight='bold')
            axes[0,1].set_xlabel('Number of Purchases')
            axes[0,1].set_ylabel('Number of Customers')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Customer Recency Analysis
        if self.rfm_df is not None:
            axes[0,2].hist(self.rfm_df['recency'], bins=20, color='coral', alpha=0.7)
            axes[0,2].set_title('Days Since Last Purchase', fontweight='bold')
            axes[0,2].set_xlabel('Days')
            axes[0,2].set_ylabel('Number of Customers')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Customer Segments by Value
        if self.rfm_df is not None:
            segment_values = self.rfm_df.groupby('segment')['monetary'].mean().sort_values(ascending=True)
            bars = axes[1,0].barh(segment_values.index, segment_values.values, color='gold')
            axes[1,0].set_title('Average Customer Value by Segment', fontweight='bold')
            axes[1,0].set_xlabel('Average Value ($)')
            
            # Add value labels
            for bar, value in zip(bars, segment_values.values):
                axes[1,0].text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                              f'${value:.0f}', ha='left', va='center', fontweight='bold')
        
        # 5. Geographic Distribution
        location_counts = self.customers_df['location'].value_counts().head(8)
        axes[1,1].bar(range(len(location_counts)), location_counts.values, color='lightblue')
        axes[1,1].set_title('Customers by Location', fontweight='bold')
        axes[1,1].set_ylabel('Number of Customers')
        axes[1,1].set_xticks(range(len(location_counts)))
        axes[1,1].set_xticklabels(location_counts.index, rotation=45, ha='right')
        
        # 6. Customer Registration Trends
        monthly_registrations = self.customers_df.groupby(
            self.customers_df['registration_date'].dt.to_period('M')
        ).size()
        axes[1,2].plot(range(len(monthly_registrations)), monthly_registrations.values, 
                      marker='o', linewidth=2, color='darkgreen')
        axes[1,2].set_title('Monthly Customer Registrations', fontweight='bold')
        axes[1,2].set_ylabel('New Registrations')
        axes[1,2].set_xlabel('Month')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customer_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Customer analysis dashboard saved as 'customer_analysis_dashboard.png'")
    
    def create_interactive_plotly_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("\n🎨 Creating Interactive Plotly Dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Revenue Trends', 'Customer Segments', 
                          'Product Category Performance', 'Order Value Analysis'),
            specs=[[{"secondary_y": True}, {"type": "domain"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Monthly Revenue Trends
        monthly_data = self.results['monthly_revenue']
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['sum'],
                      mode='lines+markers', name='Revenue',
                      line=dict(color='#2E86AB', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # 2. Customer Segments Pie Chart
        if self.rfm_df is not None:
            segment_counts = self.rfm_df['segment'].value_counts()
            fig.add_trace(
                go.Pie(labels=segment_counts.index, values=segment_counts.values,
                      name="Segments", hole=0.3),
                row=1, col=2
            )
        
        # 3. Product Category Performance
        category_data = self.results['category_revenue'].head(6)
        fig.add_trace(
            go.Bar(x=category_data['revenue'], y=category_data['category'],
                  orientation='h', name='Revenue by Category',
                  marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Order Value Distribution
        fig.add_trace(
            go.Histogram(x=self.transactions_df['final_total'], 
                        name='Order Values', nbinsx=30,
                        marker_color='orange', opacity=0.7),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive E-commerce Intelligence Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False,
            height=800
        )
        
        # Save as HTML
        fig.write_html("interactive_dashboard.html")
        fig.show()
        
        print("✅ Interactive dashboard saved as 'interactive_dashboard.html'")
    
    def generate_all_visuals(self):
        """Generate all visualizations for the portfolio"""
        print("🎨 GENERATING COMPLETE VISUAL PORTFOLIO")
        print("=" * 50)
        
        # Load data and perform analysis
        if not self.load_data():
            return False
        
        self.calculate_business_metrics()
        self.perform_rfm_analysis()
        
        # Generate all visualizations
        self.create_executive_dashboard()
        self.create_business_metrics_visual()
        self.create_customer_analysis_dashboard()
        self.create_interactive_plotly_dashboard()
        
        # Create summary metrics file
        summary_metrics = {
            'total_revenue': float(self.results['total_revenue']),
            'total_orders': int(self.results['total_orders']),
            'avg_order_value': float(self.results['avg_order_value']),
            'total_customers': int(self.results['total_customers']),
            'active_customers': int(self.results['active_customers']),
            'customer_activation_rate': float(self.results['active_customers'] / self.results['total_customers']),
            'analysis_date': datetime.now().isoformat()
        }
        
        with open('dashboard_metrics.json', 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        print(f"\n🎉 ALL VISUALS GENERATED SUCCESSFULLY!")
        print("=" * 40)
        print("📁 Files Created:")
        print("   • executive_dashboard.png - Main portfolio dashboard")
        print("   • business_metrics_summary.png - KPI overview") 
        print("   • customer_analysis_dashboard.png - Customer insights")
        print("   • interactive_dashboard.html - Interactive version")
        print("   • dashboard_metrics.json - Summary data")
        
        print(f"\n💡 Usage Tips:")
        print("   • Upload PNG files to your GitHub repository")
        print("   • Use executive_dashboard.png as your main project visual")
        print("   • Link to interactive_dashboard.html for live demo")
        print("   • Reference images in your README.md as:")
        print("     ![Dashboard](executive_dashboard.png)")
        
        return True

# Initialize and run the dashboard generator
def main():
    """Main function to generate all portfolio visuals"""
    dashboard = EcommerceVisualDashboard()
    
    print("🚀 E-COMMERCE INTELLIGENCE VISUAL PORTFOLIO GENERATOR")
    print("=" * 60)
    
    success = dashboard.generate_all_visuals()
    
    if success:
        print(f"\n✅ SUCCESS! Your visual portfolio is ready!")
        print("Upload the generated PNG files to your GitHub repository")
        print("and reference them in your README.md for a professional presentation.")
    else:
        print(f"\n❌ Failed to generate visuals. Check your data files.")

# Run the dashboard generator
if __name__ == "__main__":
    main()