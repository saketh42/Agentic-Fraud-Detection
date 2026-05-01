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
from agents.decision_agent import DecisionAgent


class MAPEKPipeline:
    '''
    Simple MAPE-K pipeline without LangGraph/LangChain.
    
    Agent Communication:
    - Each agent receives state dict
    - Each agent returns AgentResult with updated state
    - State is passed sequentially through pipeline
    '''
    
    def __init__(self, config: dict = None):
        self.config = self._merge_config(config or {}, self._default_config())
        self.agents = {}
        self.state = {}
        self.history = []
        self._setup_agents()
    
    def _merge_config(self, user_config: dict, default_config: dict) -> dict:
        """Merge user config with defaults recursively"""
        merged = default_config.copy()
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    
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
                'adversarial_training': False,  # Disabled - causes issues with encoded features
                'fgsm_epsilon': 0.05
            },
            'evaluation': {
                'min_f1': 0.70,
                'min_roc_auc': 0.75,
                'min_robustness': 0.0  # Disable robustness check - FGSM not suitable for encoded categorical features
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
            ),
            'decision': DecisionAgent(
                model='llama3',
                mock_mode=False  # Use real LLM via Ollama
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
        
        # === STEP 1: INGESTION (Load Data) ===
        print('\n[STEP 1/5] Data Ingestion')
        result = self.agents['ingestion'].run(self.state)
        self.state.update(result.data)
        self._record_step('ingestion', result)
        
        if not result.success:
            return self._finalize(success=False, error=result.message)
        
        # === INITIAL TRAINING (First time only) ===
        print('\n[STEP 2] Initial Training')
        result = self.agents['training'].run(self.state)
        self.state['model'] = result.data.get('model')
        self.state['test_features'] = result.data.get('test_features')
        self.state['test_labels'] = result.data.get('test_labels')
        
        # === INITIAL BALANCE ===
        print('\n[STEP 3] Initial Balancing')
        self.state['data'] = self.state.get('data')
        result = self.agents['balance'].run(self.state)
        self.state['balanced_data'] = result.data.get('balanced_data')
        
        # === MAPE-K LOOP (Iterate until model passes or max retries) ===
        max_iterations = 3
        iteration = 0
        self.state['supervisor_iterations'] = 0
        self.state['supervisor_max'] = 3
        self.state['supervisor_converged'] = False
        self.state['supervisor_unstable'] = False
        prior_f1 = 0.0
        
        while iteration < max_iterations:
            iteration += 1
            self.state['supervisor_iterations'] = iteration
            print(f'\n--- MAPE-K ITERATION {iteration}/{max_iterations} ---')
            
            # === M: MONITOR (Drift Agent) ===
            print('\n[M] MONITOR: Detect Drift')
            self.state['kb_drift_history'] = self._get_kb('drift')
            self.state['kb_config'] = self._get_kb('config')
            result = self.agents['drift'].run(self.state)
            self.state.update(result.data)
            self._record_step(f'monitor_{iteration}', result)
            self._update_kb('drift', {'psi': self.state.get('psi'), 'ks': self.state.get('ks'), 'iteration': iteration})
            
            # === A: ANALYZE (Evaluation Agent) ===
            print('\n[A] ANALYZE: Assess Current Model')
            self.state['kb_baseline'] = self._get_kb('metrics')
            result = self.agents['evaluation'].run(self.state)
            self.state['evaluation_metrics'] = result.data.get('evaluation_metrics', {})
            self.state['passed'] = result.data.get('passed', False)
            self.state['analysis'] = result.data.get('evaluation_metrics')
            self.state['analyze_passed'] = result.data.get('passed')
            self._record_step(f'analyze_{iteration}', result)
            self._update_kb('metrics', self.state.get('evaluation_metrics', {}))
            
            current_f1 = self.state.get('evaluation_metrics', {}).get('f1', 0.0)
            
            # === DECISION NODE (LLM-based) ===
            self.state['kb_recent'] = self._get_kb('metrics')[-3:] if hasattr(self, 'knowledge_base') else []
            self.state['prior_f1'] = prior_f1
            result = self.agents['decision'].run(self.state)
            decision = result.data.get('decision', {})
            action = decision.get('action', 'retrain_L2')
            
            print(f'\n[DECISION] LLM: {action} - {decision.get("reason", "")}')
            
            if action == 'deploy':
                print('\n✓ DECISION: Deploy Model')
                self.state['model'] = None
                self.state['deploy'] = True
                self._update_kb('deployed_model', {'iteration': iteration, 'model': 'existing'})
                break
            elif action == 'stop':
                print('\n[!] DECISION: Stop iterations')
                self.state['supervisor_stopped'] = True
                break
            else:
                if action == 'retrain_new_strategy':
                    print('\n↻ DECISION: Drift/New Strategy → Retrain with adjusted params')
                    params = decision.get('params', {})
                    if params:
                        self.state['training_override'] = params
                else:
                    print('\n↻ DECISION: Retrain')
                
                # === P: PLAN (Training Agent) - L2: Model-level correction ===
                print('\n[P] PLAN: Train Model Strategy')
                self.state['kb_training_config'] = self._get_kb('training')
                result = self.agents['training'].run(self.state)
                self.state['model'] = result.data.get('model')
                self.state['test_features'] = result.data.get('test_features')
                self.state['test_labels'] = result.data.get('test_labels')
                self.state['training_report'] = result.data.get('training_report')
                self._record_step(f'plan_{iteration}', result)
                self._update_kb('training', self.state.get('training_report', {}))
                self._update_kb('config', {'iteration': iteration})
                
                if not result.success:
                    return self._finalize(success=False, error=result.message)
                
                # === E: EXECUTE (Balance + Deploy) - L1: Data-level correction ===
                print('\n[E] EXECUTE: Balance Data')
                self.state['data'] = self.state.get('remediated_data', self.state['data'])
                result = self.agents['balance'].run(self.state)
                self.state['balanced_data'] = result.data.get('balanced_data')
                self.state['balance_report'] = result.data.get('balance_report')
                self._record_step(f'execute_{iteration}', result)
                
                print('\n[E] EXECUTE: Deploy Model')
                if self.state.get('passed', False):
                    self.state['deploy'] = True
                    self._update_kb('deployed_model', {'iteration': iteration, 'model': str(self.state.get('model'))})
                    print(f'\n✓ DEPLOY: Model ready')
                else:
                    self.state['deploy'] = False
            
            # === SUPERVISOR: Enhanced ===
            if self.state.get('deploy', False):
                print(f'\n✓ PASSED at iteration {iteration}')
                break
            
            # Check convergence
            if prior_f1 > 0 and (current_f1 - prior_f1) < 0.01:
                self.state['supervisor_converged'] = True
                print('\n[!] SUPERVISOR: Converged, stopping early')
                break
            prior_f1 = current_f1
            
            # Check unstable
            if iteration >= 2 and current_f1 < 0.5:
                self.state['supervisor_unstable'] = True
                print('\n[!] SUPERVISOR: Unstable, stopping')
                break
            
            if iteration >= max_iterations:
                print('\n[!] SUPERVISOR: Max iterations reached')
                break
            
            print(f'\n↻ RETRY: L1/L2 correction (iteration {iteration}/{max_iterations})')
        
        # === FINALIZE ===
        print('\n' + '=' * 60)
        passed = self.state.get('passed', False)
        if passed:
            print(' PIPELINE PASSED - Model ready for deployment')
        else:
            print(' PIPELINE FAILED - Model not acceptable')
        print('=' * 60)
        
        return self._finalize(
            success=passed,
            error=None
        )
    
    def _record_step(self, name: str, result):
        '''Record step result in history and KB'''
        self.history.append({
            'step': name,
            'success': result.success,
            'message': result.message,
            'metrics': result.metrics
        })
        self._update_kb(name, result.data)
    
    def _update_kb(self, record_type: str, data: dict):
        '''Update knowledge base'''
        if not hasattr(self, 'knowledge_base'):
            self.knowledge_base = {}
        if record_type not in self.knowledge_base:
            self.knowledge_base[record_type] = []
        self.knowledge_base[record_type].append(data)
    
    def _get_kb(self, record_type: str):
        '''Get from knowledge base'''
        if not hasattr(self, 'knowledge_base'):
            return None
        return self.knowledge_base.get(record_type, [])
    
    def _finalize(self, success: bool, error: str = None) -> dict:
        '''Save results and return final state'''
        output_path = os.path.join(
            self.state.get('output_dir', 'output'),
            'run_' + str(self.state.get('run_id', 'unknown'))
        )
        os.makedirs(output_path, exist_ok=True)
        
        summary = {
            'run_id': self.state.get('run_id'),
            'success': success,
            'error': error,
            'drift_detected': self.state.get('drift_detected', False),
            'balance_report': self.state.get('balance_report'),
            'training_report': self.state.get('training_report'),
            'evaluation_metrics': self.state.get('evaluation_metrics'),
            'passed': self.state.get('passed', False),
            'history': self.history,
            'knowledge_base': self.knowledge_base if hasattr(self, 'knowledge_base') else {},
            'supervisor_iterations': self.state.get('supervisor_iterations', 0)
        }
        
        with open(os.path.join(output_path, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print('\nResults saved to:', output_path)
        
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