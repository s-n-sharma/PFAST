import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import os

class DataDisplay:
    """Handles the display and visualization of PFAS and water site data."""
    
    def __init__(self):
        """Initialize the data display module."""
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def display_search_results(self, zipcode: str, nearest_sites: List[Dict], 
                             location_info: Optional[Dict] = None) -> None:
        """Display search results for a zipcode.
        
        Args:
            zipcode: The searched zipcode
            nearest_sites: List of nearest water sites
            location_info: Optional location information
        """
        print("\n" + "="*80)
        print(f"PFAS WATER SITE SEARCH RESULTS FOR ZIPCODE: {zipcode}")
        print("="*80)
        
        # Display location information
        if location_info:
            print(f"\n📍 Location: {location_info.get('address', 'Unknown')}")
            print(f"   Coordinates: {location_info.get('latitude', 'N/A'):.4f}, {location_info.get('longitude', 'N/A'):.4f}")
        
        # Display search summary
        print(f"\n🔍 Found {len(nearest_sites)} water sites within search radius")
        
        if not nearest_sites:
            print("\n❌ No water sites found within the specified radius.")
            print("   Try increasing the search radius or checking the zipcode.")
            return
        
        # Display sites
        print(f"\n📊 WATER SITES WITH PFAS DATA:")
        print("-" * 80)
        
        for i, site in enumerate(nearest_sites, 1):
            self._display_site_info(site, i)
        
        # Display summary statistics
        self._display_summary_statistics(nearest_sites)
    
    def _display_site_info(self, site: Dict, index: int) -> None:
        """Display information for a single water site.
        
        Args:
            site: Water site dictionary
            index: Site index number
        """
        print(f"\n{id}. {site['name']} ({site['type']})")
        print(f"   📍 Location: {site['city']}, {site['state']} ({site['zipcode']})")
        print(f"   📏 Distance: {site.get('distance_miles', 'N/A')} miles")
        print(f"   🧪 Total PFAS: {site['total_pfas_ppb']} ppb")
        print(f"   📅 Last Tested: {site['last_tested']}")
        print(f"   ⚠️  Status: {site['status']}")
        
        # Display PFAS compounds if present
        if site['pfas_compounds']:
            print(f"   🔬 PFAS Compounds Detected:")
            for compound_id in site['pfas_compounds']:
                print(f"      • {compound_id}")
        else:
            print(f"   ✅ No PFAS compounds detected")
    
    def _display_summary_statistics(self, sites: List[Dict]) -> None:
        """Display summary statistics for the found sites.
        
        Args:
            sites: List of water site dictionaries
        """
        if not sites:
            return
            
        print(f"\n📈 SUMMARY STATISTICS:")
        print("-" * 40)
        
        # Calculate statistics
        total_pfas_levels = [site['total_pfas_ppb'] for site in sites]
        status_counts = {}
        for site in sites:
            status = site['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"   Average PFAS Level: {sum(total_pfas_levels)/len(total_pfas_levels):.1f} ppb")
        print(f"   Maximum PFAS Level: {max(total_pfas_levels):.1f} ppb")
        print(f"   Minimum PFAS Level: {min(total_pfas_levels):.1f} ppb")
        
        print(f"\n   Status Distribution:")
        for status, count in status_counts.items():
            print(f"      {status}: {count} sites")
    
    def display_pfas_compound_details(self, compound_info: Dict) -> None:
        """Display detailed information about a PFAS compound.
        
        Args:
            compound_info: Dictionary containing compound information
        """
        print(f"\n🔬 PFAS COMPOUND DETAILS:")
        print("-" * 50)
        print(f"   Chemical ID: {compound_info.get('Chemical ID', 'N/A')}")
        print(f"   Name: {compound_info.get('IUPACName', 'N/A')}")
        print(f"   Formula: {compound_info.get('MolecularFormula', 'N/A')}")
        print(f"   Molecular Weight: {compound_info.get('MolecularWeight', 'N/A')} g/mol")
        print(f"   LogP: {compound_info.get('LogP', 'N/A')}")
        print(f"   Boiling Point: {compound_info.get('Boiling Point', 'N/A')}")
        print(f"   Melting Point: {compound_info.get('Melting Point', 'N/A')}")
    
    def create_distance_chart(self, sites: List[Dict], save_path: Optional[str] = None) -> None:
        """Create a bar chart showing PFAS levels vs distance.
        
        Args:
            sites: List of water site dictionaries
            save_path: Optional path to save the chart
        """
        if not sites:
            print("No sites to plot")
            return
            
        # Prepare data
        distances = [site.get('distance_miles', 0) for site in sites]
        pfas_levels = [site['total_pfas_ppb'] for site in sites]
        names = [site['name'] for site in sites]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(range(len(sites)), pfas_levels, 
                      color=['red' if level > 50 else 'orange' if level > 20 else 'green' 
                             for level in pfas_levels])
        
        plt.xlabel('Water Sites')
        plt.ylabel('PFAS Level (ppb)')
        plt.title('PFAS Levels in Nearby Water Sites')
        plt.xticks(range(len(sites)), [f"{name[:20]}..." if len(name) > 20 else name 
                                      for name in names], rotation=45, ha='right')
        
        # Add distance labels
        for i, (bar, distance) in enumerate(zip(bars, distances)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{distance:.1f} mi', ha='center', va='bottom', fontsize=8)
        
        # Add threshold lines
        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate Level (20 ppb)')
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='High Level (50 ppb)')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_map(self, sites: List[Dict], target_zipcode: str, 
                              target_coords: Optional[tuple] = None) -> None:
        """Create an interactive map showing water sites.
        
        Args:
            sites: List of water site dictionaries
            target_zipcode: Target zipcode
            target_coords: Target coordinates (lat, lon)
        """
        if not sites:
            print("No sites to map")
            return
            
        # Create map
        fig = go.Figure()
        
        # Add water sites
        for site in sites:
            # Determine marker color based on PFAS level
            if site['total_pfas_ppb'] > 50:
                color = 'red'
            elif site['total_pfas_ppb'] > 20:
                color = 'orange'
            else:
                color = 'green'
            
            # Add site marker
            fig.add_trace(go.Scattermapbox(
                lat=[site['latitude']],
                lon=[site['longitude']],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=15,
                    color=color,
                    opacity=0.8
                ),
                text=f"{site['name']}<br>PFAS: {site['total_pfas_ppb']} ppb<br>Distance: {site.get('distance_miles', 'N/A')} mi<br>Status: {site['status']}",
                hoverinfo='text',
                name=site['name']
            ))
        
        # Add target location if coordinates provided
        if target_coords:
            fig.add_trace(go.Scattermapbox(
                lat=[target_coords[0]],
                lon=[target_coords[1]],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=20,
                    color='blue',
                    symbol='star'
                ),
                text=f"Target Location<br>Zipcode: {target_zipcode}",
                hoverinfo='text',
                name='Target Location'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Water Sites Near {target_zipcode}',
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=sum(site['latitude'] for site in sites) / len(sites),
                    lon=sum(site['longitude'] for site in sites) / len(sites)
                ),
                zoom=8
            ),
            height=600,
            showlegend=True
        )
        
        # Show the map
        fig.show()
    
    def display_help(self) -> None:
        """Display help information for the system."""
        print("\n" + "="*80)
        print("PFAS WATER SITE LOCATOR - HELP")
        print("="*80)
        print("\nThis system helps you find water sites with PFAS contamination near a zipcode.")
        print("\n📋 COMMANDS:")
        print("   search <zipcode> [radius] - Search for water sites near a zipcode")
        print("   help                      - Show this help message")
        print("   quit                      - Exit the program")
        print("\n📊 UNDERSTANDING THE RESULTS:")
        print("   • PFAS levels are measured in parts per billion (ppb)")
        print("   • Status levels: Safe (<10 ppb), Low (10-20 ppb), Moderate (20-50 ppb), High (>50 ppb)")
        print("   • Distance is calculated in miles from your zipcode")
        print("\n⚠️  IMPORTANT NOTES:")
        print("   • This system uses sample data for demonstration")
        print("   • Always verify information with local authorities")
        print("   • PFAS levels can change over time")
        print("\n🔬 PFAS COMPOUNDS:")
        print("   • PFOA (335-67-1): Perfluorooctanoic acid")
        print("   • PFOS (1763-23-1): Perfluorooctane sulfonic acid")
        print("   • Other PFAS compounds may also be present")
        print("="*80) 