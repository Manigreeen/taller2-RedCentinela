from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def _minimax_value(self, state: GameState, agent_index: int, depth_left: int):
      
      self.nodes_evaluated += 1
      
      if state.is_win() or state.is_lose() or depth_left == 0:
        return evaluation_function(state)
      
      legal_actions = state.get_legal_actions(agent_index)
      siguiente_agente = (agent_index + 1) % state.get_num_agents()
      
      minimax_values = []
      for action in legal_actions:
        minimax_values.append(self._minimax_value(state.generate_successor(agent_index, action), siguiente_agente, depth_left - 1))
      
      if agent_index == 0:
        return max(minimax_values)
      else:
        return min(minimax_values)
    
    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        
        self.nodes_evaluated = 0
        self.nodes_evaluated += 1
        
        legal_actions = state.get_legal_actions(0)
        if legal_actions == []:
          return None
        
        action_minimax = 0
        action_chosen = ""
        for action in legal_actions:
          successor_state = state.generate_successor(0, action)
          successor_minimax = self._minimax_value(successor_state, 1, self.depth - 1)
          
          if successor_minimax > action_minimax or action_chosen == "":
            action_minimax = successor_minimax
            action_chosen = action
        
        return action_chosen
      
class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""
        
    def _alphabeta_value(self, state: GameState, agent_index: int, depth_left: int, alpha: float, beta: float):
      self.nodes_evaluated += 1
        
      if state.is_win() or state.is_lose() or depth_left == 0:
        return evaluation_function(state)
          
      legal_actions = state.get_legal_actions(agent_index)
      siguiente_agente = (agent_index + 1) % state.get_num_agents()
      
      if agent_index == 0:
        value = float("-inf")
        for action in legal_actions:
          sucesor = state.generate_successor(agent_index, action)
          child_value = self._alphabeta_value(sucesor, siguiente_agente, depth_left - 1, alpha, beta)
          value = max(value, child_value)
          
          if value >= beta:
            return value
          
          alpha = max(alpha, value)
        return value
      else:
        value = float("inf")
        for action in legal_actions:
          sucesor = state.generate_successor(agent_index, action)
          child_value = self._alphabeta_value(sucesor, siguiente_agente, depth_left - 1, alpha, beta)
          value = min(value, child_value)
          
          if value <= alpha:
            return value
          
          beta = min(beta, value)
        return value
      
    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        
        self.nodes_evaluated = 0
        self.nodes_evaluated += 1
               
        legal_actions = state.get_legal_actions(0)
        if legal_actions == []:
          return None
               
        best_value = float("-inf")
        best_action = ""
        alpha = float("-inf")
        beta = float("inf")
                   
        for action in legal_actions:
         successor_state = state.generate_successor(0, action)
         value = self._alphabeta_value(successor_state, 1, self.depth - 1, alpha, beta)
              
         if value > best_value or best_action == "":
           best_value = value
           best_action = action
                 
         alpha = max(alpha, best_value)
                  
        return best_action