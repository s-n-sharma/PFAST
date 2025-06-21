import requests
import json
from typing import Tuple, Optional, List, Dict
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

class GeolocationService:
    """Service for handling geolocation operations and distance calculations."""
    
    def __init__(self):
        """Initialize the geolocation service."""
        self.geolocator = Nominatim(user_agent="pfas_locator")
        self.cache = {}
    
    def zipcode_to_coordinates(self, zipcode: str) -> Optional[Tuple[float, float]]:
        """Convert a zipcode to latitude and longitude coordinates.
        
        Args:
            zipcode: The zipcode to convert
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # Check cache first
        if zipcode in self.cache:
            return self.cache[zipcode]
        
        try:
            # Try to geocode the zipcode
            location = self.geolocator.geocode(f"{zipcode}, USA")
            
            if location:
                coords = (location.latitude, location.longitude)
                self.cache[zipcode] = coords
                return coords
            else:
                print(f"Could not find coordinates for zipcode: {zipcode}")
                return None
                
        except Exception as e:
            print(f"Error geocoding zipcode {zipcode}: {e}")
            return None
    
    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate the distance between two coordinates in miles.
        
        Args:
            coord1: First coordinate tuple (lat, lon)
            coord2: Second coordinate tuple (lat, lon)
            
        Returns:
            Distance in miles
        """
        try:
            distance = geodesic(coord1, coord2).miles
            return round(distance, 2)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0.0
    
    def find_nearest_sites(self, target_zipcode: str, water_sites: List[Dict], max_distance: float = 100.0) -> List[Dict]:
        """Find the nearest water sites to a given zipcode.
        
        Args:
            target_zipcode: The zipcode to search from
            water_sites: List of water site dictionaries
            max_distance: Maximum distance in miles to include sites
            
        Returns:
            List of water sites with distance information, sorted by distance
        """
        target_coords = self.zipcode_to_coordinates(target_zipcode)
        
        if not target_coords:
            print(f"Could not find coordinates for zipcode: {target_zipcode}")
            return []
        
        sites_with_distance = []
        
        for site in water_sites:
            site_coords = (site['latitude'], site['longitude'])
            distance = self.calculate_distance(target_coords, site_coords)
            
            if distance <= max_distance:
                site_copy = site.copy()
                site_copy['distance_miles'] = distance
                sites_with_distance.append(site_copy)
        
        # Sort by distance
        sites_with_distance.sort(key=lambda x: x['distance_miles'])
        
        return sites_with_distance
    
    def get_location_info(self, zipcode: str) -> Optional[Dict]:
        """Get detailed location information for a zipcode.
        
        Args:
            zipcode: The zipcode to get information for
            
        Returns:
            Dictionary containing location information or None if not found
        """
        try:
            location = self.geolocator.geocode(f"{zipcode}, USA")
            
            if location:
                return {
                    'zipcode': zipcode,
                    'address': location.address,
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'raw': location.raw
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error getting location info for {zipcode}: {e}")
            return None
    
    def validate_zipcode(self, zipcode: str) -> bool:
        """Validate if a zipcode is in correct format.
        
        Args:
            zipcode: The zipcode to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic US zipcode validation (5 digits or 5+4 format)
        import re
        pattern = r'^\d{5}(-\d{4})?$'
        return bool(re.match(pattern, zipcode))
    
    def get_search_radius_suggestions(self, zipcode: str) -> Dict[str, float]:
        """Get suggested search radii based on location type.
        
        Args:
            zipcode: The zipcode to get suggestions for
            
        Returns:
            Dictionary of radius suggestions in miles
        """
        # This could be enhanced with population density data
        return {
            'local': 10.0,      # Local area
            'regional': 50.0,   # Regional area
            'state': 100.0,     # State-wide
            'national': 500.0   # National
        } 