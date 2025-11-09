from typing import Dict, List, Optional, Callable
from threading import Lock, Event
from queue import PriorityQueue
import time
from dataclasses import dataclass
from enum import Enum

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskContext:
    task_id: str
    priority: TaskPriority
    state: TaskState
    start_time: float
    callback: Callable
    args: tuple = ()
    kwargs: dict = None
    pause_time: Optional[float] = None
    error: Optional[str] = None

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskContext] = {}
        self.task_queue = PriorityQueue()
        self.lock = Lock()
        self.stop_event = Event()
        
    def add_task(self, task_id: str, callback: Callable, priority: TaskPriority = TaskPriority.NORMAL,
                 *args, **kwargs) -> bool:
        """Add a new task to the manager."""
        with self.lock:
            if task_id in self.tasks:
                return False
            
            task = TaskContext(
                task_id=task_id,
                priority=priority,
                state=TaskState.PENDING,
                start_time=time.time(),
                callback=callback,
                args=args,
                kwargs=kwargs or {}
            )
            
            self.tasks[task_id] = task
            self.task_queue.put((-priority.value, task_id))
            return True
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a running task."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.state == TaskState.RUNNING:
                task.state = TaskState.PAUSED
                task.pause_time = time.time()
                return True
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.state == TaskState.PAUSED:
                task.state = TaskState.RUNNING
                task.pause_time = None
                self.task_queue.put((-task.priority.value, task_id))
                return True
            return False
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.state = TaskState.COMPLETED
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed with an error message."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.state = TaskState.FAILED
            task.error = error
            return True
    
    def get_next_task(self) -> Optional[TaskContext]:
        """Get the next task to execute based on priority."""
        try:
            while not self.stop_event.is_set():
                _, task_id = self.task_queue.get(timeout=1)
                with self.lock:
                    task = self.tasks.get(task_id)
                    if task and task.state in [TaskState.PENDING, TaskState.PAUSED]:
                        task.state = TaskState.RUNNING
                        return task
        except:
            return None
    
    def get_active_tasks(self) -> List[TaskContext]:
        """Get all currently active tasks."""
        with self.lock:
            return [task for task in self.tasks.values() 
                   if task.state in [TaskState.RUNNING, TaskState.PAUSED]]
    
    def stop_all(self):
        """Stop all tasks and clear the queue."""
        self.stop_event.set()
        with self.lock:
            for task in self.tasks.values():
                if task.state in [TaskState.RUNNING, TaskState.PAUSED]:
                    task.state = TaskState.COMPLETED
            
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except:
                    break
    
    def remove_completed(self):
        """Remove all completed or failed tasks."""
        with self.lock:
            completed_tasks = [task_id for task_id, task in self.tasks.items()
                             if task.state in [TaskState.COMPLETED, TaskState.FAILED]]
            for task_id in completed_tasks:
                del self.tasks[task_id]