import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os

class PFASDataProcessor:

    water_sites_to_zipcode = { \
    "City of Pleasonton": ['94566', '94588'], \
    'East Bay Municipal Utility District': ['94501', '94502', '94601', '94602', '94603', '94605', \
        '94606', '94607', '94608', '94609', '94610', '94611', '94612', '94618', '94619', '94621', '94701', \
            '94702', '94703', '94704', '94705', '94706', '94707', '94708', '94709', '94710', '94525', '94530', \
                '94563', '94564', '94580', '94801', '94803', '94804', '94805', '94806'],
    'Alameda County Water District': ['94536', '94537', '94538', '94539', '94555', '94560', '94587'], \
    'San Jose Water Company': ['95008', '95011', '95030', '95032', '95033', '95070', '95110', '95111', '95112', \
        '95113', '95116', '95117', '95118', '95119', '95120', '95121', '95122', '95123', '95124', '95125', \
            '95126', '95127', '95128', '95129', '95130', '95131', '95132', '95133', '95134', '95135', '95136',  
            '95138', '95139', '95148']}
    
    
    """Processes PFAS chemical data and water site information."""
    
    def __init__(self, pfas_data_path: str = "../datagen/pfas_property_data.csv"):
        """Initialize the PFAS data processor.
        
        Args:
            pfas_data_path: Path to the PFAS properties CSV file
        """
        self.pfas_data_path = pfas_data_path
        self.pfas_data = None
        self.water_sites = None
        self.load_data()
    
    def load_data(self):
        """Load PFAS chemical data and create sample water sites."""
        try:
            # Load PFAS chemical properties
            self.pfas_data = pd.read_csv(self.pfas_data_path)
            print(f"Loaded {len(self.pfas_data)} PFAS chemicals")
            
            # Create sample water sites with PFAS data
            self.create_sample_water_sites()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create minimal sample data if file not found
            self.create_minimal_sample_data()
    
    def create_sample_water_sites(self):
        """Create sample water sites with realistic PFAS contamination data."""
        # Sample water sites across different regions
        sample_sites = [
            {
                'site_id': 'WS001',
                'name': 'Central Water Treatment Plant',
                'type': 'Treatment Plant',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'zipcode': '10001',
                'city': 'New York',
                'state': 'NY',
                'pfas_compounds': ['335-67-1', '1763-23-1', '375-73-5'],
                'total_pfas_ppb': 45.2,
                'last_tested': '2024-01-15',
                'status': 'Elevated'
            },
            {
                'site_id': 'WS002',
                'name': 'Riverside Drinking Water Facility',
                'type': 'Drinking Water',
                'latitude': 34.0522,
                'longitude': -118.2437,
                'zipcode': '90210',
                'city': 'Beverly Hills',
                'state': 'CA',
                'pfas_compounds': ['335-67-1', '375-95-1'],
                'total_pfas_ppb': 12.8,
                'last_tested': '2024-02-20',
                'status': 'Moderate'
            },
            {
                'site_id': 'WS003',
                'name': 'Industrial Zone Groundwater Well',
                'type': 'Groundwater',
                'latitude': 41.8781,
                'longitude': -87.6298,
                'zipcode': '60601',
                'city': 'Chicago',
                'state': 'IL',
                'pfas_compounds': ['1763-23-1', '375-73-5', '355-46-4', '375-95-1'],
                'total_pfas_ppb': 89.5,
                'last_tested': '2024-01-30',
                'status': 'High'
            },
            {
                'site_id': 'WS004',
                'name': 'Municipal Water Supply Station',
                'type': 'Municipal Supply',
                'latitude': 29.7604,
                'longitude': -95.3698,
                'zipcode': '77001',
                'city': 'Houston',
                'state': 'TX',
                'pfas_compounds': ['335-67-1'],
                'total_pfas_ppb': 8.3,
                'last_tested': '2024-02-10',
                'status': 'Low'
            },
            {
                'site_id': 'WS005',
                'name': 'Residential Area Water Tower',
                'type': 'Water Tower',
                'latitude': 33.7490,
                'longitude': -84.3880,
                'zipcode': '30301',
                'city': 'Atlanta',
                'state': 'GA',
                'pfas_compounds': ['1763-23-1', '375-73-5'],
                'total_pfas_ppb': 23.7,
                'last_tested': '2024-02-05',
                'status': 'Moderate'
            },
            {
                'site_id': 'WS006',
                'name': 'Coastal Water Treatment Center',
                'type': 'Treatment Plant',
                'latitude': 25.7617,
                'longitude': -80.1918,
                'zipcode': '33101',
                'city': 'Miami',
                'state': 'FL',
                'pfas_compounds': ['335-67-1', '375-95-1', '335-76-2'],
                'total_pfas_ppb': 67.2,
                'last_tested': '2024-01-25',
                'status': 'High'
            },
            {
                'site_id': 'WS007',
                'name': 'Mountain Spring Water Source',
                'type': 'Spring Water',
                'latitude': 39.7392,
                'longitude': -104.9903,
                'zipcode': '80201',
                'city': 'Denver',
                'state': 'CO',
                'pfas_compounds': [],
                'total_pfas_ppb': 0.5,
                'last_tested': '2024-02-15',
                'status': 'Safe'
            },
            {
                'site_id': 'WS008',
                'name': 'University Campus Water System',
                'type': 'Campus Supply',
                'latitude': 42.3601,
                'longitude': -71.0589,
                'zipcode': '02101',
                'city': 'Boston',
                'state': 'MA',
                'pfas_compounds': ['335-67-1', '1763-23-1'],
                'total_pfas_ppb': 15.9,
                'last_tested': '2024-02-12',
                'status': 'Moderate'
            }
        ]
        
        self.water_sites = pd.DataFrame(sample_sites)
        print(f"Created {len(self.water_sites)} sample water sites")
    
    def create_minimal_sample_data(self):
        """Create minimal sample data if main data file is not available."""
        # Create minimal PFAS data
        minimal_pfas = [
            {
                'Chemical ID': '335-67-1',
                'IUPACName': '2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-pentadecafluorooctanoic acid',
                'MolecularWeight': 414.07,
                'MolecularFormula': 'C8HF15O2',
                'LogP': 6.3,
                'Boiling Point': '192 °C',
                'Melting Point': '54.3 °C'
            },
            {
                'Chemical ID': '1763-23-1',
                'IUPACName': '1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-heptadecafluorooctane-1-sulfonic acid',
                'MolecularWeight': 500.13,
                'MolecularFormula': 'C8HF17O3S',
                'LogP': 5.0,
                'Boiling Point': '249 °C',
                'Melting Point': 'Not Available'
            }
        ]
        
        self.pfas_data = pd.DataFrame(minimal_pfas)
        self.create_sample_water_sites()
    
    def get_pfas_compound_info(self, chemical_id: str) -> Optional[Dict]:
        """Get detailed information about a specific PFAS compound.
        
        Args:
            chemical_id: The chemical ID of the PFAS compound
            
        Returns:
            Dictionary containing compound information or None if not found
        """
        if self.pfas_data is None:
            return None
            
        compound = self.pfas_data[self.pfas_data['Chemical ID'] == chemical_id]
        if len(compound) == 0:
            return None
            
        return compound.iloc[0].to_dict()
    
    def get_water_sites_by_zipcode(self, zipcode: str) -> pd.DataFrame:
        """Get water sites for a specific zipcode.
        
        Args:
            zipcode: The zipcode to search for
            
        Returns:
            DataFrame containing water sites in the zipcode
        """
        if self.water_sites is None:
            return pd.DataFrame()
            
        return self.water_sites[self.water_sites['zipcode'] == zipcode]
    
    def get_all_water_sites(self) -> pd.DataFrame:
        """Get all water sites.
        
        Returns:
            DataFrame containing all water sites
        """
        return self.water_sites if self.water_sites is not None else pd.DataFrame()
    
    def get_pfas_summary(self) -> Dict:
        """Get a summary of PFAS data.
        
        Returns:
            Dictionary containing PFAS data summary
        """
        if self.pfas_data is None:
            return {}
            
        return {
            'total_compounds': len(self.pfas_data),
            'compounds_with_logp': len(self.pfas_data[self.pfas_data['LogP'].notna()]),
            'compounds_with_boiling_point': len(self.pfas_data[self.pfas_data['Boiling Point'].notna()]),
            'average_molecular_weight': self.pfas_data['MolecularWeight'].mean() if 'MolecularWeight' in self.pfas_data.columns else 0
        }
    
    def get_contamination_summary(self) -> Dict:
        """Get a summary of water contamination data.
        
        Returns:
            Dictionary containing contamination summary
        """
        if self.water_sites is None:
            return {}
            
        status_counts = self.water_sites['status'].value_counts().to_dict()
        avg_pfas = self.water_sites['total_pfas_ppb'].mean()
        max_pfas = self.water_sites['total_pfas_ppb'].max()
        
        return {
            'total_sites': len(self.water_sites),
            'status_distribution': status_counts,
            'average_pfas_level': avg_pfas,
            'maximum_pfas_level': max_pfas,
            'sites_with_pfas': len(self.water_sites[self.water_sites['total_pfas_ppb'] > 0])
        } 