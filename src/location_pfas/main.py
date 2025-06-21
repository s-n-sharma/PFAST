#!/usr/bin/env python3
"""
PFAS Water Site Locator with Safety Prediction
A system to find water sites with PFAS contamination and predict safety using ML.
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
from safety_integration import SafetyIntegration
from enhanced_display import EnhancedDataDisplay

class PFASLocator:
    """Main class for the PFAS water site locator system with safety prediction."""
    
    def __init__(self):
        """Initialize the PFAS locator system."""
        print("Initializing PFAS Water Site Locator with Safety Prediction...")
        
        # Initialize components
        self.data_processor = PFASDataProcessor()
        self.geo_service = GeolocationService()
        self.display = DataDisplay()
        self.safety_integration = SafetyIntegration()
        self.enhanced_display = EnhancedDataDisplay()
        
        print("✅ System initialized successfully!")
    
    def search_water_sites(self, zipcode: str, radius: float = 100.0, 
                          include_safety_analysis: bool = True, 
                          include_visualizations: bool = True) -> None:
        """Search for water sites near a zipcode with optional safety analysis.
        
        Args:
            zipcode: The zipcode to search from
            radius: Search radius in miles
            include_safety_analysis: Whether to include ML safety analysis
            include_visualizations: Whether to create visualizations
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
        
        if not nearest_sites:
            print(f"❌ No water sites found within {radius} miles of {zipcode}")
            return
        
        # Perform safety analysis if requested
        if include_safety_analysis:
            print(f"\n🧠 Performing ML safety analysis...")
            analyzed_sites = self.safety_integration.analyze_water_site_safety(nearest_sites)
            safety_report = self.safety_integration.generate_safety_report(analyzed_sites)
            
            # Display enhanced results with safety analysis
            self.enhanced_display.display_safety_analysis(analyzed_sites, safety_report)
            
            # Create visualizations if requested
            if include_visualizations:
                try:
                    self.enhanced_display.create_safety_visualizations(analyzed_sites, safety_report)
                    self.enhanced_display.create_risk_heatmap(analyzed_sites)
                    self.enhanced_display.create_safety_comparison_chart(analyzed_sites)
                except Exception as e:
                    print(f"⚠️  Could not create some visualizations: {e}")
            
            # Save safety report
            try:
                self.enhanced_display.create_safety_report_file(analyzed_sites, safety_report)
            except Exception as e:
                print(f"⚠️  Could not save safety report: {e}")
        else:
            # Display basic results without safety analysis
            self.display.display_search_results(zipcode, nearest_sites, location_info)
            
            # Create basic visualizations if requested
            if include_visualizations:
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
    
    def show_safety_model_info(self) -> None:
        """Show information about the safety prediction model."""
        print(f"\n🤖 SAFETY PREDICTION MODEL INFORMATION:")
        print("-" * 50)
        
        # Train model if not already trained
        if not self.safety_integration.is_model_trained:
            print("Training safety prediction model...")
            self.safety_integration.train_safety_model()
        
        # Get model information
        model_info = self.safety_integration.safety_predictor.get_model_info()
        self.enhanced_display.display_model_performance(model_info)
    
    def train_safety_model(self, force_retrain: bool = False) -> None:
        """Train the safety prediction model.
        
        Args:
            force_retrain: Whether to force retraining
        """
        print(f"\n🧠 Training safety prediction model...")
        self.safety_integration.train_safety_model(force_retrain)
        print("✅ Safety model training completed!")
    
    def interactive_mode(self) -> None:
        """Run the system in interactive mode."""
        print("\n" + "="*80)
        print("PFAS WATER SITE LOCATOR WITH SAFETY PREDICTION - INTERACTIVE MODE")
        print("="*80)
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n🔍 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif command == 'help':
                    self._display_interactive_help()
                
                elif command == 'summary':
                    self.show_pfas_summary()
                
                elif command == 'model_info':
                    self.show_safety_model_info()
                
                elif command == 'train_model':
                    self.train_safety_model()
                
                elif command.startswith('search'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("❌ Usage: search <zipcode> [radius] [--no-safety] [--no-visualizations]")
                        continue
                    
                    zipcode = parts[1]
                    radius = float(parts[2]) if len(parts) > 2 else 100.0
                    include_safety = '--no-safety' not in parts
                    include_visualizations = '--no-visualizations' not in parts
                    
                    self.search_water_sites(zipcode, radius, include_safety, include_visualizations)
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _display_interactive_help(self) -> None:
        """Display help information for interactive mode."""
        print("\n" + "="*80)
        print("PFAS WATER SITE LOCATOR WITH SAFETY PREDICTION - HELP")
        print("="*80)
        print("\nThis system helps you find water sites with PFAS contamination and predict safety using ML.")
        print("\n📋 COMMANDS:")
        print("   search <zipcode> [radius] [--no-safety] [--no-visualizations] - Search for water sites near a zipcode")
        print("   summary                                    - Show PFAS data summary")
        print("   model_info                                 - Show ML model information")
        print("   train_model                                - Train the safety prediction model")
        print("   help                                       - Show this help message")
        print("   quit                                       - Exit the program")
        print("\n📊 UNDERSTANDING THE RESULTS:")
        print("   • PFAS levels are measured in parts per billion (ppb)")
        print("   • ML Safety Score: Lower values indicate safer water")
        print("   • Risk Levels: Low Risk, Moderate Risk, High Risk, Very High Risk")
        print("   • Priority Levels: Low, Medium, High, Critical")
        print("\n🤖 ML SAFETY PREDICTION:")
        print("   • Uses Multi-Layer Perceptron (MLP) neural networks")
        print("   • Predicts safety based on PFAS compound concentrations")
        print("   • Provides confidence scores and risk assessments")
        print("   • Generates automated recommendations")
        print("\n📊 VISUALIZATIONS:")
        print("   • Interactive charts are saved as HTML files")
        print("   • Use --no-visualizations for terminal-only output")
        print("   • Open HTML files in a web browser to view charts")
        print("\n⚠️  IMPORTANT NOTES:")
        print("   • This system uses sample data for demonstration")
        print("   • Always verify information with local authorities")
        print("   • ML predictions are for educational purposes")
        print("   • Safety thresholds are based on current EPA guidelines")
        print("="*80)
    
    def run_single_search(self, zipcode: str, radius: float = 100.0, 
                         include_safety_analysis: bool = True,
                         include_visualizations: bool = True) -> None:
        """Run a single search and exit.
        
        Args:
            zipcode: The zipcode to search
            radius: Search radius in miles
            include_safety_analysis: Whether to include safety analysis
            include_visualizations: Whether to create visualizations
        """
        self.search_water_sites(zipcode, radius, include_safety_analysis, include_visualizations)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="PFAS Water Site Locator with Safety Prediction - Find water sites and predict safety using ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --zipcode 10001
  python main.py --zipcode 90210 --radius 50
  python main.py --zipcode 10001 --no-safety
  python main.py --zipcode 10001 --no-visualizations
  python main.py --interactive
  python main.py --train-model
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
        '--no-safety',
        action='store_true',
        help='Disable safety analysis (faster results)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable all visualizations (better for terminal)'
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
    
    parser.add_argument(
        '--model-info',
        action='store_true',
        help='Show ML model information'
    )
    
    parser.add_argument(
        '--train-model',
        action='store_true',
        help='Train the safety prediction model'
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
    elif args.model_info:
        locator.show_safety_model_info()
    elif args.train_model:
        locator.train_safety_model()
    elif args.zipcode:
        include_safety = not args.no_safety
        include_visualizations = not args.no_visualizations
        locator.run_single_search(args.zipcode, args.radius, include_safety, include_visualizations)
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n💡 Tip: Use --interactive for interactive mode")

if __name__ == "__main__":
    main() 