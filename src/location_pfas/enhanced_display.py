import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import os

# Try to import plotly, but make it optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available. Interactive visualizations will be skipped.")

class EnhancedDataDisplay:
    """Enhanced display module for safety predictions and analysis."""
    
    def __init__(self):
        """Initialize the enhanced display module."""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def display_safety_analysis(self, analyzed_sites: List[Dict], 
                               safety_report: Dict) -> None:
        """Display comprehensive safety analysis results.
        
        Args:
            analyzed_sites: List of analyzed water sites
            safety_report: Safety report dictionary
        """
        print("\n" + "="*100)
        print("🔬 PFAS SAFETY ANALYSIS RESULTS")
        print("="*100)
        
        # Display summary
        self._display_safety_summary(safety_report)
        
        # Display risk distribution
        self._display_risk_distribution(safety_report)
        
        # Display critical sites
        self._display_critical_sites(safety_report)
        
        # Display detailed site analysis
        self._display_detailed_site_analysis(analyzed_sites)
        
        # Display recommendations
        self._display_recommendations(safety_report)
    
    def _display_safety_summary(self, safety_report: Dict) -> None:
        """Display safety analysis summary.
        
        Args:
            safety_report: Safety report dictionary
        """
        summary = safety_report['summary']
        
        print(f"\n📊 SAFETY ANALYSIS SUMMARY:")
        print("-" * 50)
        print(f"   Total Sites Analyzed: {summary['total_sites_analyzed']}")
        print(f"   Average Safety Score: {summary['average_safety_score']:.2f}")
        print(f"   Maximum Safety Score: {summary['max_safety_score']:.2f}")
        print(f"   Average PFAS Level: {summary['average_pfas_level']:.1f} ppb")
        print(f"   Maximum PFAS Level: {summary['max_pfas_level']:.1f} ppb")
        print(f"   High-Risk Sites: {safety_report['high_risk_sites_count']}")
    
    def _display_risk_distribution(self, safety_report: Dict) -> None:
        """Display risk distribution information.
        
        Args:
            safety_report: Safety report dictionary
        """
        print(f"\n⚠️  RISK DISTRIBUTION:")
        print("-" * 30)
        
        for risk_level, count in safety_report['risk_distribution'].items():
            percentage = (count / safety_report['summary']['total_sites_analyzed']) * 100
            print(f"   {risk_level}: {count} sites ({percentage:.1f}%)")
        
        print(f"\n🎯 PRIORITY DISTRIBUTION:")
        print("-" * 30)
        
        for priority, count in safety_report['priority_distribution'].items():
            percentage = (count / safety_report['summary']['total_sites_analyzed']) * 100
            print(f"   {priority}: {count} sites ({percentage:.1f}%)")
    
    def _display_critical_sites(self, safety_report: Dict) -> None:
        """Display critical sites requiring immediate attention.
        
        Args:
            safety_report: Safety report dictionary
        """
        critical_sites = safety_report['critical_sites']
        
        if not critical_sites:
            print(f"\n✅ No critical sites identified.")
            return
        
        print(f"\n🚨 CRITICAL SITES REQUIRING IMMEDIATE ATTENTION:")
        print("-" * 60)
        
        for i, site in enumerate(critical_sites, 1):
            print(f"\n{i}. {site['site_name']}")
            print(f"   📍 Location: {site['location']}")
            print(f"   ⚠️  Risk Level: {site['risk_level']}")
            print(f"   🧪 Safety Score: {site['safety_score']:.2f}")
            print(f"   📊 Total PFAS: {site['total_pfas']:.1f} ppb")
            print(f"   🎯 Priority: {site['priority']}")
            print(f"   💡 Recommendation: {site['recommendation']}")
    
    def _display_detailed_site_analysis(self, analyzed_sites: List[Dict]) -> None:
        """Display detailed analysis for each site.
        
        Args:
            analyzed_sites: List of analyzed water sites
        """
        print(f"\n📋 DETAILED SITE ANALYSIS:")
        print("-" * 50)
        
        for i, site in enumerate(analyzed_sites, 1):
            print(f"\n{i}. {site['name']} ({site['type']})")
            print(f"   📍 Location: {site['city']}, {site['state']} ({site['zipcode']})")
            print(f"   📏 Distance: {site.get('distance_miles', 'N/A')} miles")
            print(f"   🧪 Total PFAS: {site['total_pfas_ppb']} ppb")
            print(f"   🤖 ML Safety Category: {site['ml_safety_category']}")
            print(f"   📊 ML Safety Score: {site['ml_safety_score']:.2f}")
            print(f"   ⚠️  ML Risk Level: {site['ml_risk_level']}")
            print(f"   🎯 Priority: {site['safety_analysis']['priority_level']}")
            print(f"   📈 Confidence: {site['safety_analysis']['confidence_score']:.1%}")
            
            # Display risk factors
            if site['safety_analysis']['risk_factors']:
                print(f"   🔍 Risk Factors:")
                for factor in site['safety_analysis']['risk_factors']:
                    print(f"      • {factor}")
            
            # Display comparative risk
            comp_risk = site['safety_analysis']['comparative_risk']
            print(f"   📊 Comparative Risk: {comp_risk['percentile']}")
            print(f"   📈 Relative Risk: {comp_risk['relative_risk']:.1f}x baseline")
            
            print(f"   💡 Recommendation: {site['ml_recommendation']}")
    
    def _display_recommendations(self, safety_report: Dict) -> None:
        """Display recommendations from the safety analysis.
        
        Args:
            safety_report: Safety report dictionary
        """
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        for i, recommendation in enumerate(safety_report['recommendations'], 1):
            print(f"   {i}. {recommendation}")
    
    def create_safety_visualizations(self, analyzed_sites: List[Dict], 
                                   safety_report: Dict) -> None:
        """Create comprehensive safety visualizations.
        
        Args:
            analyzed_sites: List of analyzed water sites
            safety_report: Safety report dictionary
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Skipping interactive visualizations (Plotly not available)")
            return
        
        try:
            print(f"\n📊 Creating safety visualizations...")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Safety Score Distribution', 'PFAS Levels vs Safety Score',
                              'Risk Level Distribution', 'Priority Level Distribution'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # 1. Safety Score Distribution
            safety_scores = [site['ml_safety_score'] for site in analyzed_sites]
            fig.add_trace(
                go.Histogram(x=safety_scores, name='Safety Scores', nbinsx=20),
                row=1, col=1
            )
            
            # 2. PFAS Levels vs Safety Score
            pfas_levels = [site['total_pfas_ppb'] for site in analyzed_sites]
            risk_colors = []
            for site in analyzed_sites:
                if site['ml_risk_level'] == 'Very High Risk':
                    risk_colors.append('red')
                elif site['ml_risk_level'] == 'High Risk':
                    risk_colors.append('orange')
                elif site['ml_risk_level'] == 'Moderate Risk':
                    risk_colors.append('yellow')
                else:
                    risk_colors.append('green')
            
            fig.add_trace(
                go.Scatter(x=pfas_levels, y=safety_scores, mode='markers',
                          marker=dict(color=risk_colors, size=10),
                          name='Sites', text=[site['name'] for site in analyzed_sites]),
                row=1, col=2
            )
            
            # 3. Risk Level Distribution
            risk_counts = safety_report['risk_distribution']
            fig.add_trace(
                go.Bar(x=list(risk_counts.keys()), y=list(risk_counts.values()),
                      name='Risk Distribution'),
                row=2, col=1
            )
            
            # 4. Priority Level Distribution
            priority_counts = safety_report['priority_distribution']
            fig.add_trace(
                go.Bar(x=list(priority_counts.keys()), y=list(priority_counts.values()),
                      name='Priority Distribution'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='PFAS Safety Analysis Dashboard',
                height=800,
                showlegend=False
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Safety Score", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text="PFAS Level (ppb)", row=1, col=2)
            fig.update_yaxes(title_text="Safety Score", row=1, col=2)
            fig.update_xaxes(title_text="Risk Level", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_xaxes(title_text="Priority Level", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            
            # Save the plot as HTML file instead of showing
            fig.write_html("safety_analysis_dashboard.html")
            print("✅ Safety dashboard saved as 'safety_analysis_dashboard.html'")
            
        except Exception as e:
            print(f"⚠️  Could not create safety visualizations: {e}")
    
    def create_risk_heatmap(self, analyzed_sites: List[Dict]) -> None:
        """Create a risk heatmap visualization.
        
        Args:
            analyzed_sites: List of analyzed water sites
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Skipping risk heatmap (Plotly not available)")
            return
        
        try:
            # Prepare data for heatmap
            site_names = [site['name'][:20] for site in analyzed_sites]
            safety_scores = [site['ml_safety_score'] for site in analyzed_sites]
            pfas_levels = [site['total_pfas_ppb'] for site in analyzed_sites]
            
            # Create heatmap data
            heatmap_data = np.array([safety_scores, pfas_levels]).T
            
            # Create the heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=['Safety Score', 'PFAS Level (ppb)'],
                y=site_names,
                colorscale='RdYlGn_r',  # Red to Green (reversed for safety)
                text=[[f"{score:.2f}", f"{level:.1f}"] for score, level in zip(safety_scores, pfas_levels)],
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Water Site Risk Heatmap',
                xaxis_title='Metrics',
                yaxis_title='Water Sites',
                height=600
            )
            
            # Save as HTML file
            fig.write_html("risk_heatmap.html")
            print("✅ Risk heatmap saved as 'risk_heatmap.html'")
            
        except Exception as e:
            print(f"⚠️  Could not create risk heatmap: {e}")
    
    def create_safety_comparison_chart(self, analyzed_sites: List[Dict]) -> None:
        """Create a comparison chart showing different safety metrics.
        
        Args:
            analyzed_sites: List of analyzed water sites
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Skipping safety comparison chart (Plotly not available)")
            return
        
        try:
            # Prepare data
            site_names = [site['name'][:15] for site in analyzed_sites]
            safety_scores = [site['ml_safety_score'] for site in analyzed_sites]
            pfas_levels = [site['total_pfas_ppb'] for site in analyzed_sites]
            confidence_scores = [site['safety_analysis']['confidence_score'] for site in analyzed_sites]
            
            # Create the comparison chart
            fig = go.Figure()
            
            # Add safety scores
            fig.add_trace(go.Bar(
                name='Safety Score',
                x=site_names,
                y=safety_scores,
                yaxis='y',
                offsetgroup=1
            ))
            
            # Add PFAS levels (scaled)
            scaled_pfas = [level / max(pfas_levels) * max(safety_scores) for level in pfas_levels]
            fig.add_trace(go.Bar(
                name='PFAS Level (scaled)',
                x=site_names,
                y=scaled_pfas,
                yaxis='y',
                offsetgroup=2
            ))
            
            # Add confidence scores
            scaled_confidence = [conf * max(safety_scores) for conf in confidence_scores]
            fig.add_trace(go.Scatter(
                name='Confidence (scaled)',
                x=site_names,
                y=scaled_confidence,
                mode='lines+markers',
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title='Safety Metrics Comparison',
                xaxis_title='Water Sites',
                yaxis_title='Safety Score / Scaled PFAS Level',
                yaxis2=dict(
                    title='Confidence Score',
                    overlaying='y',
                    side='right'
                ),
                barmode='group',
                height=600
            )
            
            # Save as HTML file
            fig.write_html("safety_comparison.html")
            print("✅ Safety comparison chart saved as 'safety_comparison.html'")
            
        except Exception as e:
            print(f"⚠️  Could not create safety comparison chart: {e}")
    
    def display_model_performance(self, model_info: Dict) -> None:
        """Display ML model performance information.
        
        Args:
            model_info: Model information dictionary
        """
        print(f"\n🤖 ML MODEL INFORMATION:")
        print("-" * 40)
        print(f"   Status: {model_info['status']}")
        
        if model_info['status'] == 'Trained':
            print(f"   Feature Count: {model_info['feature_count']}")
            print(f"   Classification Classes: {', '.join(model_info['classification_classes'])}")
            print(f"   Classification Layers: {model_info['classification_layers']}")
            print(f"   Regression Layers: {model_info['regression_layers']}")
            
            print(f"\n   Features Used:")
            for feature in model_info['feature_names']:
                print(f"      • {feature}")
    
    def create_safety_report_file(self, analyzed_sites: List[Dict], 
                                 safety_report: Dict, filename: str = "safety_report.txt") -> None:
        """Create a text file with the safety report.
        
        Args:
            analyzed_sites: List of analyzed water sites
            safety_report: Safety report dictionary
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                f.write("PFAS WATER SAFETY ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary
                summary = safety_report['summary']
                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Sites Analyzed: {summary['total_sites_analyzed']}\n")
                f.write(f"Average Safety Score: {summary['average_safety_score']:.2f}\n")
                f.write(f"Maximum Safety Score: {summary['max_safety_score']:.2f}\n")
                f.write(f"Average PFAS Level: {summary['average_pfas_level']:.1f} ppb\n")
                f.write(f"Maximum PFAS Level: {summary['max_pfas_level']:.1f} ppb\n")
                f.write(f"High-Risk Sites: {safety_report['high_risk_sites_count']}\n\n")
                
                # Critical sites
                f.write("CRITICAL SITES\n")
                f.write("-" * 20 + "\n")
                for site in safety_report['critical_sites']:
                    f.write(f"• {site['site_name']} ({site['location']})\n")
                    f.write(f"  Risk Level: {site['risk_level']}\n")
                    f.write(f"  Safety Score: {site['safety_score']:.2f}\n")
                    f.write(f"  Total PFAS: {site['total_pfas']:.1f} ppb\n")
                    f.write(f"  Priority: {site['priority']}\n")
                    f.write(f"  Recommendation: {site['recommendation']}\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(safety_report['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
            
            print(f"✅ Safety report saved to {filename}")
        except Exception as e:
            print(f"⚠️  Could not save safety report: {e}") 