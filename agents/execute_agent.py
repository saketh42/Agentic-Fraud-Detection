"""
Execute Agent - Execute phase of MAPE-K
Mock tool layer — all actions go through here. Logs to KnowledgeStore.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


MOCK_TOOLS = {
    'ALLOW_TRANSACTION': lambda tid: {'outcome': 'SUCCESS', 'details': 'Transaction allowed'},
    'TRIGGER_MFA': lambda tid: {'outcome': 'SUCCESS', 'details': 'MFA challenge sent to user'},
    'BLOCK_TRANSACTION': lambda tid: {'outcome': 'SUCCESS', 'details': 'Transaction blocked'},
    'CREATE_MANUAL_REVIEW_CASE': lambda tid: {'outcome': 'SUCCESS', 'details': f'Review case created for {tid}'},
    'RUN_DRIFT_CHECK': lambda tid: {'outcome': 'PASSED', 'details': 'No drift detected'},
    'RUN_ADVERSARIAL_SIMULATION': lambda tid: {'outcome': 'SUCCESS', 'details': 'Adversarial simulation completed'},
    'TRIGGER_RETRAINING_REVIEW': lambda tid: {'outcome': 'SUBMITTED', 'details': 'Retraining review queued'},
}


class ExecuteAgent(BaseAgent):
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("ExecuteAgent")
        self.knowledge = KnowledgeStore(db_path)

    def run(self, state: dict) -> AgentResult:
        self.log("Executing planned actions...")

        plan = state.get('plan', {})
        actions = plan.get('actions', [])
        transaction_id = state.get('masked_transaction_id', state.get('transaction_id', 'unknown'))

        execution_results = []
        all_success = True

        for action in actions:
            tool_fn = MOCK_TOOLS.get(action)
            if tool_fn:
                result = tool_fn(transaction_id)
                execution_results.append({
                    'action': action,
                    'status': result['outcome'],
                    'details': result['details']
                })
                self.knowledge.store_execution(
                    transaction_id=transaction_id,
                    action=action,
                    outcome=result['outcome'],
                    details=result['details']
                )
                self.log(f"  -> {action}: {result['outcome']}")
            else:
                execution_results.append({
                    'action': action,
                    'status': 'UNKNOWN_TOOL',
                    'details': f'No handler for {action}'
                })
                all_success = False

        execute_result = {
            'execution_results': execution_results,
            'all_actions_successful': all_success,
            'executed_actions_count': len(execution_results),
            'transaction_id': transaction_id
        }

        return AgentResult(
            success=all_success,
            data=execute_result,
            message=f"Executed {len(execution_results)} actions",
            metrics={'executed': len(execution_results), 'success': all_success}
        )
