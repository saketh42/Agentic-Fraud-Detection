#!/usr/bin/env python3
'''
Main Pipeline - Orchestrates all agents in sequence
Simple MCP-style (no LangGraph, no LangChain)
'''
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from agents.drift_agent import DriftAgent
from agents.balance_agent import BalanceAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.ingestion_agent import IngestionAgent


class MAPEKPipeline:
    '''
    Simple MAPE-K pipeline without LangGraph/LangChain.
    
    Agent Communication:
    - Each agent receives state dict
    - Each agent returns AgentResult with updated state
    - State is passed sequentially through pipeline
    '''
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.agents = {}
        self.state = {}
        self.history = []
        self._setup_agents()
    
    def _default_config(self) -> dict:
        return {
            'drift': {
                'psi_threshold': 0.20,
                'ks_threshold': 0.05
            },
            'balance': {
                'target_ratio': 0.3,
                'ctgan_epochs': 100
            },
            'training': {
                'model_type': 'gradient_boosting',
                'adversarial_training': True,
                'fgsm_epsilon': 0.05
            },
            'evaluation': {
                'min_f1': 0.70,
                'min_roc_auc': 0.75,
                'min_robustness': 0.60
            }
        }
    
    def _setup_agents(self):
        '''Initialize all agents with config'''
        self.agents = {
            'ingestion': IngestionAgent(),
            'drift': DriftAgent(
                psi_threshold=self.config['drift']['psi_threshold'],
                ks_threshold=self.config['drift']['ks_threshold']
            ),
            'balance': BalanceAgent(
                target_ratio=self.config['balance']['target_ratio'],
                ctgan_epochs=self.config['balance']['ctgan_epochs']
            ),
            'training': TrainingAgent(
                model_type=self.config['training']['model_type'],
                adversarial_training=self.config['training']['adversarial_training'],
                fgsm_epsilon=self.config['training']['fgsm_epsilon']
            ),
            'evaluation': EvaluationAgent(
                robustness_thresholds=self.config['evaluation']
            )
        }
    
    def run(self, data: pd.DataFrame, 
            target_col: str = 'is_fraud',
            feature_cols: list = None,
            output_dir: str = 'output') -> dict:
        '''
        Run the complete MAPE-K pipeline.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (auto-detected if None)
            output_dir: Directory for outputs
        
        Returns:
            Final state dict with all results
        '''
        print('=' * 60)
        print(' MAPE-K FRAUD DETECTION PIPELINE')
        print('=' * 60)
        print(f'Input data: {len(data)} rows, {len(data.columns)} columns')
        print()
        
        # Initialize state
        if feature_cols is None:
            feature_cols = [c for c in data.columns if c != target_col]
        
        self.state = {
            'data': data,
            'target_col': target_col,
            'feature_cols': feature_cols,
            'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'output_dir': output_dir
        }
        
        # === STEP 1: INGESTION (Load & Preprocess) ===
        print('\n[STEP 1/5] Data Ingestion')
        result = self.agents['ingestion'].run(self.state)
        self.state.update(result.data)
        self._record_step('ingestion', result)
        
        if not result.success:
            return self._finalize(success=False, error=result.message)
        
        # === STEP 2: DRIFT DETECTION (Monitor) ===
        print('\n[STEP 2/5] Drift Detection (M in MAPE)')
        result = self.agents['drift'].run(self.state)
        self.state.update(result.data)
        self._record_step('drift', result)
        
        if not result.success:
            return self._finalize(success=False, error=result.message)
        
        # === STEP 3: BALANCE (Execute) ===
        print('\n[STEP 3/5] Class Balancing (E in MAPE)')
        self.state['data'] = self.state.get('remediated_data', self.state['data'])
        result = self.agents['balance'].run(self.state)
        self.state['balanced_data'] = result.data.get('balanced_data')
        self.state['balance_report'] = result.data.get('balance_report')
        self._record_step('balance', result)
        
        if not result.success:
            return self._finalize(success=False, error=result.message)
        
        # === STEP 4: TRAINING (Plan) ===
        print('\n[STEP 4/5] Model Training (P in MAPE)')
        result = self.agents['training'].run(self.state)
        self.state['model'] = result.data.get('model')
        self.state['training_report'] = result.data.get('training_report')
        self._record_step('training', result)
        
        if not result.success:
            return self._finalize(success=False, error=result.message)
        
        # === STEP 5: EVALUATION (Analyze) ===
        print('\n[STEP 5/5] Model Evaluation (A in MAPE)')
        result = self.agents['evaluation'].run(self.state)
        self.state['evaluation_metrics'] = result.data.get('evaluation_metrics')
        self.state['passed'] = result.data.get('passed')
        self._record_step('evaluation', result)
        
        # === FINALIZE ===
        print('\n' + '=' * 60)
        if self.state['passed']:
            print(' PIPELINE PASSED - Model ready for deployment')
        else:
            print(' PIPELINE FAILED - Model not acceptable')
        print('=' * 60)
        
        return self._finalize(
            success=self.state['passed'],
            error=None
        )
    
    def _record_step(self, name: str, result):
        '''Record step result in history'''
        self.history.append({
            'step': name,
            'success': result.success,
            'message': result.message,
            'metrics': result.metrics
        })
    
    def _finalize(self, success: bool, error: str = None) -> dict:
        '''Save results and return final state'''
        output_path = os.path.join(
            self.state.get('output_dir', 'output'),
            f'run_{self.state.get(\"run_id\", \"unknown\")}'
        )
        os.makedirs(output_path, exist_ok=True)
        
        # Save run summary
        summary = {
            'run_id': self.state.get('run_id'),
            'success': success,
            'error': error,
            'drift_detected': self.state.get('drift_detected', False),
            'balance_report': self.state.get('balance_report'),
            'training_report': self.state.get('training_report'),
            'evaluation_metrics': self.state.get('evaluation_metrics'),
            'passed': self.state.get('passed', False),
            'history': self.history
        }
        
        with open(os.path.join(output_path, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f'\nResults saved to: {output_path}')
        
        return {
            'success': success,
            'state': self.state,
            'summary': summary
        }


def run_pipeline(data_path: str = None, 
                data: pd.DataFrame = None,
                target_col: str = 'is_fraud',
                output_dir: str = 'output') -> dict:
    '''
    Convenience function to run pipeline from file or DataFrame.
    
    Usage:
        # From file:
        result = run_pipeline('path/to/data.csv')
        
        # From DataFrame:
        result = run_pipeline(data=df, target_col='is_fraud')
    '''
    if data is None and data_path is None:
        raise ValueError('Must provide either data or data_path')
    
    if data is None:
        data = pd.read_csv(data_path)
    
    pipeline = MAPEKPipeline()
    return pipeline.run(data, target_col=target_col, output_dir=output_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MAPE-K Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--target', type=str, default='is_fraud', help='Target column name')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    result = run_pipeline(args.data, target_col=args.target, output_dir=args.output)
    print(f'\nFinal result: SUCCESS' if result['success'] else '\nFinal result: FAILED')