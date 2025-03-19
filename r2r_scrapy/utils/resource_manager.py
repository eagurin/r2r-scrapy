import psutil
import time
import logging
import threading
import asyncio
from collections import deque

class ResourceManager:
    """Manage system resources for optimal performance"""
    
    def __init__(self, settings=None):
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        self.settings = settings or {}
        
        # Resource limits
        self.max_cpu_percent = self.settings.get('MAX_CPU_PERCENT', 80)
        self.max_memory_percent = self.settings.get('MAX_MEMORY_PERCENT', 80)
        self.check_interval = self.settings.get('RESOURCE_CHECK_INTERVAL', 5)  # seconds
        
        # Task queue
        self.task_queue = deque()
        self.active_tasks = 0
        self.max_concurrent_tasks = self.settings.get('MAX_CONCURRENT_TASKS', 10)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.running:
            try:
                # Check CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Log resource usage
                self.logger.debug(f"Resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
                
                # Adjust max concurrent tasks based on resource usage
                self._adjust_concurrency(cpu_percent, memory_percent)
                
                # Process task queue
                self._process_queue()
                
                # Wait before next check
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")
                time.sleep(self.check_interval)
    
    def _adjust_concurrency(self, cpu_percent, memory_percent):
        """Adjust max concurrent tasks based on resource usage"""
        with self.lock:
            # Start with default max
            new_max = self.settings.get('MAX_CONCURRENT_TASKS', 10)
            
            # Reduce if CPU usage is high
            if cpu_percent > self.max_cpu_percent:
                cpu_factor = 1 - ((cpu_percent - self.max_cpu_percent) / (100 - self.max_cpu_percent))
                new_max = int(new_max * max(0.5, cpu_factor))
            
            # Reduce if memory usage is high
            if memory_percent > self.max_memory_percent:
                memory_factor = 1 - ((memory_percent - self.max_memory_percent) / (100 - self.max_memory_percent))
                new_max = int(new_max * max(0.5, memory_factor))
            
            # Ensure at least one task can run
            new_max = max(1, new_max)
            
            # Update max concurrent tasks if changed
            if new_max != self.max_concurrent_tasks:
                self.logger.info(f"Adjusting max concurrent tasks from {self.max_concurrent_tasks} to {new_max}")
                self.max_concurrent_tasks = new_max
    
    def _process_queue(self):
        """Process tasks in the queue"""
        with self.lock:
            # Check if we can start more tasks
            while self.task_queue and self.active_tasks < self.max_concurrent_tasks:
                # Get next task
                task, callback = self.task_queue.popleft()
                
                # Increment active tasks
                self.active_tasks += 1
                
                # Start task in a separate thread
                threading.Thread(target=self._run_task, args=(task, callback), daemon=True).start()
    
    def _run_task(self, task, callback):
        """Run a task and call the callback when done"""
        try:
            # Run the task
            if asyncio.iscoroutinefunction(task):
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(task())
                loop.close()
            else:
                result = task()
            
            # Call callback with result
            if callback:
                callback(result)
        except Exception as e:
            self.logger.error(f"Error running task: {e}")
            # Call callback with None result
            if callback:
                callback(None)
        finally:
            # Decrement active tasks
            with self.lock:
                self.active_tasks -= 1
    
    def submit_task(self, task, callback=None):
        """Submit a task to be executed when resources are available"""
        with self.lock:
            # Add task to queue
            self.task_queue.append((task, callback))
            self.logger.debug(f"Task submitted, queue size: {len(self.task_queue)}")
            
            # Try to process queue immediately
            self._process_queue()
    
    def get_stats(self):
        """Get resource manager statistics"""
        with self.lock:
            return {
                'active_tasks': self.active_tasks,
                'queued_tasks': len(self.task_queue),
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
            }
    
    def shutdown(self):
        """Shutdown the resource manager"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5) 