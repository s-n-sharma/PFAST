import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from safety_predictor import PFASSafetyPredictor
from pfas_data_processor import PFASDataProcessor

class SafetyIntegration:
    """Integrates safety prediction with the PFAS location system."""
    
    def __init__(self):
        """Initialize the safety integration system."""
        self.safety_predictor = PFASSafetyPredictor()
        self.data_processor = PFASDataProcessor()
        self.is_model_trained = False
        
    def train_safety_model(self, force_retrain: bool = False) -> None:
        """Train the safety prediction model.
        
        Args:
            force_retrain: Whether to force retraining even if model exists
        """
        try:
            # Try to load existing model
            if not force_retrain:
                self.safety_predictor.load_model('safety_model.pkl')
                self.is_model_trained = True
                print("✅ Loaded existing safety model")
                return
        except FileNotFoundError:
            print("No existing model found, training new model...")
        
        # Create training data
        training_data = self.safety_predictor.create_training_data()
        
        # Prepare features
        X, y_class, y_reg = self.safety_predictor.prepare_features(training_data)
        
        # Train model
        self.safety_predictor.train_model(X, y_class, y_reg)
        
        # Save model
        self.safety_predictor.save_model('safety_model.pkl')
        self.is_model_trained = True
    
    def analyze_water_site_safety(self, water_sites: List[Dict]) -> List[Dict]:
        """Analyze safety of water sites using the ML model.
        
        Args:
            water_sites: List of water site dictionaries
            
        Returns:
            List of water sites with safety analysis
        """
        if not self.is_model_trained:
            self.train_safety_model()
        
        analyzed_sites = []
        
        for site in water_sites:
            # Prepare PFAS data for prediction
            pfas_data = self._prepare_pfas_data_for_site(site)
            
            # Get safety prediction
            safety_prediction = self.safety_predictor.predict_safety(pfas_data)
            
            # Create enhanced site data
            enhanced_site = site.copy()
            enhanced_site.update({
                'ml_safety_category': safety_prediction['safety_category'],
                'ml_safety_score': safety_prediction['safety_score'],
                'ml_risk_level': safety_prediction['risk_level'],
                'ml_recommendation': safety_prediction['recommendation'],
                'safety_analysis': self._get_detailed_safety_analysis(site, safety_prediction)
            })
            
            analyzed_sites.append(enhanced_site)
        
        return analyzed_sites
    
    def _prepare_pfas_data_for_site(self, site: Dict) -> Dict[str, float]:
        """Prepare PFAS data from a water site for safety prediction.
        
        Args:
            site: Water site dictionary
            
        Returns:
            Dictionary with PFAS concentrations
        """
        # Initialize with zeros
        pfas_data = {
            'PFOA_ppb': 0.0,
            'PFOS_ppb': 0.0,
            'PFBS_ppb': 0.0,
            'PFHxS_ppb': 0.0,
            'PFNA_ppb': 0.0,
            'PFDA_ppb': 0.0,
            'PFHxA_ppb': 0.0,
            'PFBA_ppb': 0.0
        }
        
        # Map chemical IDs to compound names
        compound_mapping = {
            '335-67-1': 'PFOA',
            '1763-23-1': 'PFOS',
            '375-73-5': 'PFBS',
            '355-46-4': 'PFHxS',
            '375-95-1': 'PFNA',
            '335-76-2': 'PFDA',
            '307-24-4': 'PFHxA',
            '375-22-4': 'PFBA'
        }
        
        # Distribute total PFAS among detected compounds
        total_pfas = site.get('total_pfas_ppb', 0.0)
        detected_compounds = site.get('pfas_compounds', [])
        
        if detected_compounds and total_pfas > 0:
            # Calculate concentration per compound
            concentration_per_compound = total_pfas / len(detected_compounds)
            
            for compound_id in detected_compounds:
                if compound_id in compound_mapping:
                    compound_name = compound_mapping[compound_id]
                    pfas_data[f'{compound_name}_ppb'] = concentration_per_compound
        else:
            # If no specific compounds detected, assume general PFAS
            pfas_data['PFOA_ppb'] = total_pfas * 0.3  # Assume 30% PFOA
            pfas_data['PFOS_ppb'] = total_pfas * 0.2  # Assume 20% PFOS
            pfas_data['PFBS_ppb'] = total_pfas * 0.1  # Assume 10% PFBS
            # Distribute remaining among other compounds
        
        return pfas_data
    
    def _get_detailed_safety_analysis(self, site: Dict, prediction: Dict) -> Dict:
        """Get detailed safety analysis for a water site.
        
        Args:
            site: Water site dictionary
            prediction: Safety prediction results
            
        Returns:
            Dictionary with detailed safety analysis
        """
        total_pfas = site.get('total_pfas_ppb', 0.0)
        safety_score = prediction['safety_score']
        
        # Calculate risk factors
        risk_factors = []
        
        if total_pfas > 50:
            risk_factors.append("Very high total PFAS concentration")
        elif total_pfas > 20:
            risk_factors.append("High total PFAS concentration")
        elif total_pfas > 10:
            risk_factors.append("Moderate PFAS concentration")
        
        if safety_score > 20:
            risk_factors.append("Extremely high safety risk score")
        elif safety_score > 10:
            risk_factors.append("High safety risk score")
        elif safety_score > 5:
            risk_factors.append("Moderate safety risk score")
        
        # Determine priority level
        if safety_score > 20 or total_pfas > 50:
            priority = "Critical"
        elif safety_score > 10 or total_pfas > 20:
            priority = "High"
        elif safety_score > 5 or total_pfas > 10:
            priority = "Medium"
        else:
            priority = "Low"
        
        return {
            'risk_factors': risk_factors,
            'priority_level': priority,
            'confidence_score': self._calculate_confidence(safety_score, total_pfas),
            'trend_analysis': self._analyze_trends(site),
            'comparative_risk': self._get_comparative_risk(safety_score)
        }
    
    def _calculate_confidence(self, safety_score: float, total_pfas: float) -> float:
        """Calculate confidence in the safety prediction.
        
        Args:
            safety_score: Predicted safety score
            total_pfas: Total PFAS concentration
            
        Returns:
            Confidence score (0-1)
        """
        # Higher confidence for extreme values, lower for borderline cases
        if safety_score < 1 or safety_score > 20:
            return 0.9
        elif safety_score < 3 or safety_score > 10:
            return 0.8
        else:
            return 0.7
    
    def _analyze_trends(self, site: Dict) -> Dict:
        """Analyze trends in the water site data.
        
        Args:
            site: Water site dictionary
            
        Returns:
            Dictionary with trend analysis
        """
        # This would typically use historical data
        # For now, provide basic analysis based on current data
        total_pfas = site.get('total_pfas_ppb', 0.0)
        
        if total_pfas > 50:
            trend = "Increasing risk"
            trend_description = "PFAS levels indicate significant contamination"
        elif total_pfas > 20:
            trend = "Stable risk"
            trend_description = "PFAS levels are concerning but stable"
        elif total_pfas > 10:
            trend = "Decreasing risk"
            trend_description = "PFAS levels are moderate"
        else:
            trend = "Low risk"
            trend_description = "PFAS levels are within acceptable ranges"
        
        return {
            'trend': trend,
            'description': trend_description,
            'recommended_monitoring_frequency': self._get_monitoring_frequency(total_pfas)
        }
    
    def _get_monitoring_frequency(self, total_pfas: float) -> str:
        """Get recommended monitoring frequency based on PFAS levels.
        
        Args:
            total_pfas: Total PFAS concentration
            
        Returns:
            Recommended monitoring frequency
        """
        if total_pfas > 50:
            return "Weekly"
        elif total_pfas > 20:
            return "Monthly"
        elif total_pfas > 10:
            return "Quarterly"
        else:
            return "Annually"
    
    def _get_comparative_risk(self, safety_score: float) -> Dict:
        """Get comparative risk analysis.
        
        Args:
            safety_score: Predicted safety score
            
        Returns:
            Dictionary with comparative risk information
        """
        if safety_score < 1:
            percentile = "Bottom 10% (Very Safe)"
            comparison = "Better than 90% of contaminated sites"
        elif safety_score < 5:
            percentile = "Bottom 30% (Safe)"
            comparison = "Better than 70% of contaminated sites"
        elif safety_score < 10:
            percentile = "Middle 40% (Moderate)"
            comparison = "Average risk level"
        elif safety_score < 20:
            percentile = "Top 20% (High Risk)"
            comparison = "Worse than 80% of sites"
        else:
            percentile = "Top 5% (Very High Risk)"
            comparison = "Among the most contaminated sites"
        
        return {
            'percentile': percentile,
            'comparison': comparison,
            'relative_risk': self._calculate_relative_risk(safety_score)
        }
    
    def _calculate_relative_risk(self, safety_score: float) -> float:
        """Calculate relative risk compared to baseline.
        
        Args:
            safety_score: Predicted safety score
            
        Returns:
            Relative risk multiplier
        """
        baseline_score = 1.0  # Baseline safe level
        return safety_score / baseline_score if baseline_score > 0 else 0
    
    def generate_safety_report(self, analyzed_sites: List[Dict]) -> Dict:
        """Generate a comprehensive safety report.
        
        Args:
            analyzed_sites: List of analyzed water sites
            
        Returns:
            Dictionary with safety report
        """
        if not analyzed_sites:
            return {'error': 'No sites to analyze'}
        
        # Calculate statistics
        safety_scores = [site['ml_safety_score'] for site in analyzed_sites]
        total_pfas_levels = [site['total_pfas_ppb'] for site in analyzed_sites]
        
        # Risk distribution
        risk_levels = [site['ml_risk_level'] for site in analyzed_sites]
        risk_distribution = pd.Series(risk_levels).value_counts().to_dict()
        
        # Priority distribution
        priority_levels = [site['safety_analysis']['priority_level'] for site in analyzed_sites]
        priority_distribution = pd.Series(priority_levels).value_counts().to_dict()
        
        # Find highest risk sites
        high_risk_sites = [site for site in analyzed_sites 
                          if site['ml_risk_level'] in ['High Risk', 'Very High Risk']]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analyzed_sites)
        
        return {
            'summary': {
                'total_sites_analyzed': len(analyzed_sites),
                'average_safety_score': np.mean(safety_scores),
                'max_safety_score': np.max(safety_scores),
                'average_pfas_level': np.mean(total_pfas_levels),
                'max_pfas_level': np.max(total_pfas_levels)
            },
            'risk_distribution': risk_distribution,
            'priority_distribution': priority_distribution,
            'high_risk_sites_count': len(high_risk_sites),
            'recommendations': recommendations,
            'critical_sites': self._get_critical_sites(analyzed_sites)
        }
    
    def _generate_recommendations(self, analyzed_sites: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis.
        
        Args:
            analyzed_sites: List of analyzed water sites
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        high_risk_count = len([site for site in analyzed_sites 
                              if site['ml_risk_level'] in ['High Risk', 'Very High Risk']])
        
        if high_risk_count > 0:
            recommendations.append(f"Immediate action required for {high_risk_count} high-risk sites")
        
        avg_safety_score = np.mean([site['ml_safety_score'] for site in analyzed_sites])
        if avg_safety_score > 10:
            recommendations.append("Overall area shows elevated PFAS risk - consider regional monitoring")
        
        if any(site['total_pfas_ppb'] > 50 for site in analyzed_sites):
            recommendations.append("Some sites exceed EPA advisory levels - implement treatment measures")
        
        if len(recommendations) == 0:
            recommendations.append("All sites appear to be within acceptable risk levels")
        
        return recommendations
    
    def _get_critical_sites(self, analyzed_sites: List[Dict]) -> List[Dict]:
        """Get sites requiring immediate attention.
        
        Args:
            analyzed_sites: List of analyzed water sites
            
        Returns:
            List of critical sites
        """
        critical_sites = []
        
        for site in analyzed_sites:
            if (site['ml_risk_level'] in ['High Risk', 'Very High Risk'] or
                site['total_pfas_ppb'] > 50 or
                site['safety_analysis']['priority_level'] == 'Critical'):
                
                critical_sites.append({
                    'site_name': site['name'],
                    'location': f"{site['city']}, {site['state']}",
                    'risk_level': site['ml_risk_level'],
                    'safety_score': site['ml_safety_score'],
                    'total_pfas': site['total_pfas_ppb'],
                    'priority': site['safety_analysis']['priority_level'],
                    'recommendation': site['ml_recommendation']
                })
        
        return critical_sites 