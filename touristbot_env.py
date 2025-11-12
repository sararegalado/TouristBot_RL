# TouristBot Environment - Versión Básica v1.0
# Basado en Snake_env, adaptado para navegación turística

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# ======================
# CONFIGURACIÓN BÁSICA
# ======================
GRID_SIZE = 10  # Grid 10x10
CELL_SIZE = 50  # Cada celda = 50x50 px
TABLE_SIZE = GRID_SIZE * CELL_SIZE  # 500x500 px

# Colores para visualización (BGR para OpenCV)
COLORS = {
    "background": (0, 0, 0),        # Negro
    "agent": (255, 255, 255),       # Blanco (turista)
    "restaurant": (0, 165, 255),    # Naranja
    "museum": (0, 0, 255),          # Rojo
    "visited": (50, 50, 50),        # Gris oscuro
}

# Tipos de lugares disponibles
PLACE_TYPES = ["restaurant", "museum"]


class TouristBotEnv(gym.Env):
    """
    Entorno básico de TouristBot:
    - El agente (turista) debe navegar por una ciudad grid 10x10
    - Objetivo: llegar a un lugar específico (restaurant o museum)
    - Acciones: 4 direcciones (arriba, abajo, izquierda, derecha)
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_type="restaurant"):
        super(TouristBotEnv, self).__init__()
        
        # Espacio de acciones: 4 movimientos direccionales
        # 0: arriba, 1: abajo, 2: izquierda, 3: derecha
        self.action_space = spaces.Discrete(4)
        
        # Espacio de observación: [agent_x, agent_y, goal_x, goal_y, goal_type]
        # goal_type: 0=restaurant, 1=museum
        self.observation_space = spaces.Box(
            low=0,
            high=GRID_SIZE,
            shape=(5,),
            dtype=np.float32
        )
        
        # Estado del entorno
        self.agent_pos = [0, 0]  # Posición del agente [x, y]
        self.goal_type = goal_type  # Tipo de lugar objetivo
        self.goal_pos = [0, 0]  # Posición del objetivo
        self.places = {}  # Diccionario de lugares {tipo: posición}
        
        # Visualización
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Métricas
        self.steps = 0
        self.max_steps = 100
        self.total_reward = 0
        
        print("🌆 TouristBot Environment inicializado")
        print(f"   Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"   Objetivo inicial: {goal_type}")

    def reset(self, *, seed=None, options=None):
        """Reinicia el entorno al estado inicial"""
        # IMPORTANTE: Llamar a super().reset() primero para manejar la seed correctamente
        super().reset(seed=seed)
        
        # Permitir cambiar el objetivo desde options
        if options and "goal_type" in options:
            self.goal_type = options["goal_type"]
        
        # Posición inicial del agente (esquina inferior izquierda)
        self.agent_pos = [1, GRID_SIZE - 2]  # [x, y]
        
        # Generar lugares de forma aleatoria pero fija
        self._generate_places()
        
        # Establecer el objetivo según el tipo (hacer copia para evitar referencias)
        self.goal_pos = self.places[self.goal_type].copy()
        
        # Resetear métricas
        self.steps = 0
        self.total_reward = 0
        
        # Crear observación inicial
        observation = self._get_observation()
        info = {
            "goal_type": self.goal_type,
            "goal_position": self.goal_pos
        }
        
        return observation, info

    def step(self, action):
        """Ejecuta una acción y retorna el nuevo estado"""
        self.steps += 1
        
        # Guardar posición anterior
        prev_pos = self.agent_pos.copy()
        
        # Ejecutar acción
        self._take_action(action)
        
        # Calcular distancia al objetivo
        prev_distance = self._manhattan_distance(prev_pos, self.goal_pos)
        current_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        
        # Sistema de recompensas básico
        reward = 0
        terminated = False
        
        # 1. Recompensa por llegar al objetivo
        if self.agent_pos == self.goal_pos:
            reward = +10.0
            terminated = True
            print(f"🎉 ¡Objetivo alcanzado en {self.steps} pasos!")
        
        # 2. Recompensa por acercarse al objetivo
        elif current_distance < prev_distance:
            reward = +0.5
        
        # 3. Penalización por alejarse
        elif current_distance > prev_distance:
            reward = -0.5
        
        # 4. Pequeña penalización por cada paso (eficiencia)
        reward -= 0.1
        
        # 5. Truncar si excede el máximo de pasos
        truncated = self.steps >= self.max_steps
        if truncated:
            reward -= 5.0
            print(f"⏰ Tiempo agotado después de {self.steps} pasos")
        
        # Actualizar reward total
        self.total_reward += reward
        
        # Observación y información
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "distance_to_goal": current_distance,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renderiza el entorno visualmente"""
        # Limpiar imagen
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Dibujar grid
        for i in range(GRID_SIZE + 1):
            # Líneas verticales
            cv2.line(self.img, (i * CELL_SIZE, 0), (i * CELL_SIZE, TABLE_SIZE), (30, 30, 30), 1)
            # Líneas horizontales
            cv2.line(self.img, (0, i * CELL_SIZE), (TABLE_SIZE, i * CELL_SIZE), (30, 30, 30), 1)
        
        # Dibujar lugares
        for place_type, pos in self.places.items():
            color = COLORS[place_type]
            x, y = pos
            cv2.rectangle(
                self.img,
                (x * CELL_SIZE + 5, y * CELL_SIZE + 5),
                ((x + 1) * CELL_SIZE - 5, (y + 1) * CELL_SIZE - 5),
                color,
                -1
            )
            # Añadir etiqueta
            label = place_type[:4].upper()
            cv2.putText(
                self.img,
                label,
                (x * CELL_SIZE + 8, y * CELL_SIZE + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        # Dibujar agente (turista) - círculo blanco
        agent_x, agent_y = self.agent_pos
        center = (agent_x * CELL_SIZE + CELL_SIZE // 2, agent_y * CELL_SIZE + CELL_SIZE // 2)
        cv2.circle(self.img, center, 15, COLORS["agent"], -1)
        cv2.circle(self.img, center, 15, (100, 100, 100), 2)  # Borde
        
        # Añadir información en pantalla
        info_text = [
            f"Goal: {self.goal_type.upper()}",
            f"Steps: {self.steps}/{self.max_steps}",
            f"Reward: {self.total_reward:.1f}",
            f"Pos: ({agent_x}, {agent_y})"
        ]
        
        y_offset = 20
        for text in info_text:
            cv2.putText(
                self.img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        # Mostrar ventana
        cv2.imshow('TouristBot', self.img)
        cv2.waitKey(1)

    def close(self):
        """Cierra las ventanas de visualización"""
        cv2.destroyAllWindows()

    # ==================
    # MÉTODOS PRIVADOS
    # ==================
    
    def _generate_places(self):
        """Genera lugares en posiciones aleatorias del grid usando el RNG del entorno"""
        self.places = {}
        
        # Restaurant en la mitad superior derecha
        # Usamos self.np_random que es el RNG oficial de Gymnasium
        # integers(low, high) genera números en [low, high)
        self.places["restaurant"] = [
            int(self.np_random.integers(GRID_SIZE // 2, GRID_SIZE)),
            int(self.np_random.integers(1, GRID_SIZE // 2 + 1))
        ]
        
        # Museum en la mitad superior izquierda
        self.places["museum"] = [
            int(self.np_random.integers(1, GRID_SIZE // 2 + 1)),
            int(self.np_random.integers(1, GRID_SIZE // 2 + 1))
        ]
        
        # Asegurar que no están en la misma posición
        max_attempts = 10
        attempts = 0
        while self.places["restaurant"] == self.places["museum"] and attempts < max_attempts:
            self.places["museum"] = [
                int(self.np_random.integers(1, GRID_SIZE // 2 + 1)),
                int(self.np_random.integers(1, GRID_SIZE // 2 + 1))
            ]
            attempts += 1

    def _take_action(self, action):
        """
        Ejecuta la acción de movimiento
        0: arriba, 1: abajo, 2: izquierda, 3: derecha
        """
        x, y = self.agent_pos
        
        if action == 0:  # Arriba
            y = max(0, y - 1)
        elif action == 1:  # Abajo
            y = min(GRID_SIZE - 1, y + 1)
        elif action == 2:  # Izquierda
            x = max(0, x - 1)
        elif action == 3:  # Derecha
            x = min(GRID_SIZE - 1, x + 1)
        
        self.agent_pos = [x, y]

    def _get_observation(self):
        """
        Crea el vector de observación:
        [agent_x, agent_y, goal_x, goal_y, goal_type_encoded]
        """
        goal_type_encoded = PLACE_TYPES.index(self.goal_type)
        
        observation = np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
            goal_type_encoded
        ], dtype=np.float32)
        
        return observation

    def _manhattan_distance(self, pos1, pos2):
        """Calcula la distancia de Manhattan entre dos posiciones"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ==================
# FUNCIÓN DE PRUEBA
# ==================
def test_environment():
    """Función para probar el entorno con acciones aleatorias"""
    print("="*60)
    print("PROBANDO TOURISTBOT ENVIRONMENT")
    print("="*60)
    
    # Crear entorno
    env = TouristBotEnv(goal_type="restaurant")
    
    # Reset
    observation, info = env.reset()
    print(f"\n📍 Estado inicial:")
    print(f"   Agente en: {env.agent_pos}")
    print(f"   Objetivo: {info['goal_type']} en {info['goal_position']}")
    print(f"   Observación: {observation}")
    
    # Ejecutar algunos pasos
    print(f"\n🎮 Ejecutando acciones aleatorias...")
    
    for episode in range(3):
        observation, info = env.reset(options={"goal_type": random.choice(PLACE_TYPES)})
        print(f"\n--- Episodio {episode + 1} ---")
        print(f"Objetivo: {env.goal_type}")
        
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()  # Acción aleatoria
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            time.sleep(0.1)  # Pausa para visualización
        
        print(f"Resultado: {'✅ Éxito' if terminated else '❌ Tiempo agotado'}")
        print(f"Reward total: {env.total_reward:.2f}")
        time.sleep(1)
    
    env.close()
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_environment()
