from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import time
from threading import Lock

class TaskStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class Task:
    name: str
    status: TaskStatus
    start_time: float
    pause_time: Optional[float] = None
    metadata: Dict = None

class SystemState:
    def __init__(self):
        self.lock = Lock()
        self.active_tasks: Dict[str, Task] = {}
        self.current_mode: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.active_functions: Set[str] = set()
        self.interrupted_tasks: List[Task] = []
        
    def start_task(self, task_name: str, metadata: Dict = None) -> bool:
        with self.lock:
            if task_name in self.active_tasks:
                return False
            
            self.active_tasks[task_name] = Task(
                name=task_name,
                status=TaskStatus.RUNNING,
                start_time=time.time(),
                metadata=metadata or {}
            )
            return True
    
    def pause_task(self, task_name: str) -> bool:
        with self.lock:
            if task_name not in self.active_tasks:
                return False
            
            task = self.active_tasks[task_name]
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                task.pause_time = time.time()
                return True
            return False
    
    def resume_task(self, task_name: str) -> bool:
        with self.lock:
            if task_name not in self.active_tasks:
                return False
            
            task = self.active_tasks[task_name]
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.RUNNING
                task.pause_time = None
                return True
            return False
    
    def stop_task(self, task_name: str) -> bool:
        with self.lock:
            if task_name not in self.active_tasks:
                return False
            
            task = self.active_tasks[task_name]
            task.status = TaskStatus.STOPPED
            self.active_tasks.pop(task_name)
            return True
    
    def get_active_tasks(self) -> List[Task]:
        with self.lock:
            return [task for task in self.active_tasks.values() 
                   if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED)]
    
    def add_to_history(self, speaker: str, text: str, metadata: Dict = None):
        with self.lock:
            self.conversation_history.append({
                'timestamp': time.time(),
                'speaker': speaker,
                'text': text,
                'metadata': metadata or {}
            })
            # Keep last 50 interactions
            if len(self.conversation_history) > 50:
                self.conversation_history.pop(0)
    
    def get_recent_context(self, n: int = 5) -> List[Dict]:
        with self.lock:
            return self.conversation_history[-n:]
    
    def set_mode(self, mode: str):
        with self.lock:
            self.current_mode = mode
    
    def get_mode(self) -> Optional[str]:
        with self.lock:
            return self.current_mode
    
    def is_task_active(self, task_name: str) -> bool:
        with self.lock:
            return (task_name in self.active_tasks and 
                   self.active_tasks[task_name].status == TaskStatus.RUNNING)
    
    def interrupt_task(self, task_name: str) -> bool:
        with self.lock:
            if not self.is_task_active(task_name):
                return False
            
            task = self.active_tasks[task_name]
            self.interrupted_tasks.append(task)
            return self.pause_task(task_name)
    
    def resume_last_interrupted(self) -> Optional[str]:
        with self.lock:
            if not self.interrupted_tasks:
                return None
            
            task = self.interrupted_tasks.pop()
            if self.resume_task(task.name):
                return task.name
            return None
    
    def clear_all_tasks(self):
        with self.lock:
            self.active_tasks.clear()
            self.interrupted_tasks.clear()
            self.active_functions.clear()