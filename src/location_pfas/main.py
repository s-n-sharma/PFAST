#!/usr/bin/env python3
"""
PFAS Water Site Locator
A system to find water sites with PFAS contamination near a given zipcode.
"""

import sys
import os
import argparse
from typing import Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pfas_data_processor import PFASDataProcessor
from geolocation_service import GeolocationService
from data_display import DataDisplay

class PFASLocator:
    """Main class for the PFAS water site locator system."""
    
    def __init__(self):
        """Initialize the PFAS locator system."""
        print("Initializing PFAS Water Site Locator...")
        
        # Initialize components
        self.data_processor = PFASDataProcessor()
        self.geo_service = GeolocationService()
        self.display = DataDisplay()
        
        print("✅ System initialized successfully!")
    
    def search_water_sites(self, zipcode: str, radius: float = 100.0) -> None:
        """Search for water sites near a zipcode.
        
        Args:
            zipcode: The zipcode to search from
            radius: Search radius in miles
        """
        print(f"\n🔍 Searching for water sites near {zipcode} (radius: {radius} miles)...")
        
        # Validate zipcode
        if not self.geo_service.validate_zipcode(zipcode):
            print(f"❌ Invalid zipcode format: {zipcode}")
            print("   Please use a valid US zipcode (e.g., 10001 or 10001-1234)")
            return
        
        # Get location information
        location_info = self.geo_service.get_location_info(zipcode)
        
        # Get all water sites
        all_sites = self.data_processor.get_all_water_sites()
        if all_sites.empty:
            print("❌ No water sites data available")
            return
        
        # Convert to list of dictionaries
        sites_list = all_sites.to_dict('records')
        
        # Find nearest sites
        nearest_sites = self.geo_service.find_nearest_sites(zipcode, sites_list, radius)
        
        # Display results
        self.display.display_search_results(zipcode, nearest_sites, location_info)
        
        # Create visualizations if sites found
        if nearest_sites:
            print(f"\n📊 Creating visualizations...")
            
            # Create distance chart
            try:
                self.display.create_distance_chart(nearest_sites)
            except Exception as e:
                print(f"⚠️  Could not create chart: {e}")
            
            # Create interactive map
            try:
                target_coords = self.geo_service.zipcode_to_coordinates(zipcode)
                self.display.create_interactive_map(nearest_sites, zipcode, target_coords)
            except Exception as e:
                print(f"⚠️  Could not create map: {e}")
    
    def show_pfas_summary(self) -> None:
        """Show a summary of PFAS data."""
        print(f"\n📊 PFAS DATA SUMMARY:")
        print("-" * 40)
        
        pfas_summary = self.data_processor.get_pfas_summary()
        contamination_summary = self.data_processor.get_contamination_summary()
        
        print(f"   Total PFAS Compounds: {pfas_summary.get('total_compounds', 0)}")
        print(f"   Compounds with LogP data: {pfas_summary.get('compounds_with_logp', 0)}")
        print(f"   Average Molecular Weight: {pfas_summary.get('average_molecular_weight', 0):.1f} g/mol")
        
        print(f"\n   Total Water Sites: {contamination_summary.get('total_sites', 0)}")
        print(f"   Sites with PFAS: {contamination_summary.get('sites_with_pfas', 0)}")
        print(f"   Average PFAS Level: {contamination_summary.get('average_pfas_level', 0):.1f} ppb")
        print(f"   Maximum PFAS Level: {contamination_summary.get('maximum_pfas_level', 0):.1f} ppb")
    
    def interactive_mode(self) -> None:
        """Run the system in interactive mode."""
        print("\n" + "="*80)
        print("PFAS WATER SITE LOCATOR - INTERACTIVE MODE")
        print("="*80)
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n🔍 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif command == 'help':
                    self.display.display_help()
                
                elif command == 'summary':
                    self.show_pfas_summary()
                
                elif command.startswith('search'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("❌ Usage: search <zipcode> [radius]")
                        continue
                    
                    zipcode = parts[1]
                    radius = float(parts[2]) if len(parts) > 2 else 100.0
                    
                    self.search_water_sites(zipcode, radius)
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_single_search(self, zipcode: str, radius: float = 100.0) -> None:
        """Run a single search and exit.
        
        Args:
            zipcode: The zipcode to search
            radius: Search radius in miles
        """
        self.search_water_sites(zipcode, radius)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="PFAS Water Site Locator - Find water sites with PFAS contamination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --zipcode 10001
  python main.py --zipcode 90210 --radius 50
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        '--zipcode', '-z',
        type=str,
        help='Zipcode to search for water sites'
    )
    
    parser.add_argument(
        '--radius', '-r',
        type=float,
        default=100.0,
        help='Search radius in miles (default: 100)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Show PFAS data summary'
    )
    
    args = parser.parse_args()
    
    # Initialize the system
    try:
        locator = PFASLocator()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Run based on arguments
    if args.interactive:
        locator.interactive_mode()
    elif args.summary:
        locator.show_pfas_summary()
    elif args.zipcode:
        locator.run_single_search(args.zipcode, args.radius)
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n💡 Tip: Use --interactive for interactive mode")

if __name__ == "__main__":
    main() 