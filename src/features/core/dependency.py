"""
Dependency graph resolution for feature computation.

Provides topological sorting and cycle detection for feature dependencies.

Features:
- Topological sort for determining computation order
- Circular dependency detection with cycle reporting
- Missing dependency validation
- Graph traversal utilities
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from .errors import (
    CircularDependencyError,
    MissingDependencyError,
)


class DependencyGraph:
    """
    Manages feature dependencies and provides resolution algorithms.
    
    A dependency graph tracks which features depend on which other features,
    enables topological sorting for computation order, and detects circular
    dependencies.
    """
    
    def __init__(self):
        """Initialize an empty dependency graph."""
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.all_features: Set[str] = set()
    
    def add_feature(self, feature_name: str, dependencies: List[str]) -> None:
        """
        Add a feature and its dependencies to the graph.
        
        Args:
            feature_name: Name of the feature
            dependencies: List of feature names this feature depends on
        """
        self.all_features.add(feature_name)
        self.graph[feature_name] = set(dependencies)
        
        # Ensure all dependencies are registered as features
        for dep in dependencies:
            if dep not in self.all_features:
                self.all_features.add(dep)
    
    def get_dependencies(self, feature_name: str) -> Set[str]:
        """
        Get direct dependencies of a feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            Set of feature names this feature directly depends on
        """
        return self.graph.get(feature_name, set()).copy()
    
    def get_all_dependencies(self, feature_name: str) -> Set[str]:
        """
        Get all transitive dependencies of a feature (direct and indirect).
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            Set of all feature names this feature depends on (directly or indirectly)
        """
        all_deps = set()
        stack = [feature_name]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            deps = self.get_dependencies(current)
            all_deps.update(deps)
            stack.extend(deps)
        
        return all_deps
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect all circular dependencies in the graph.
        
        Returns:
            List of cycles, where each cycle is a list of feature names
        """
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> None:
            """Depth-first search to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start_idx = path.index(neighbor)
                    cycle = path[cycle_start_idx:] + [neighbor]
                    # Only add if not already found
                    if cycle not in cycles:
                        cycles.append(cycle)
        
        for feature in self.all_features:
            if feature not in visited:
                dfs(feature, [])
        
        return cycles
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the dependency graph.
        
        Checks for:
        1. Circular dependencies
        2. Missing dependencies
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            for cycle in cycles:
                errors.append(f"Circular dependency: {' -> '.join(cycle)}")
        
        # Check for missing dependencies
        for feature in self.all_features:
            deps = self.get_dependencies(feature)
            for dep in deps:
                if dep not in self.all_features:
                    errors.append(
                        f"Feature '{feature}' depends on missing feature '{dep}'"
                    )
        
        return len(errors) == 0, errors
    
    def topological_sort(self) -> List[str]:
        """
        Compute topological order of features for computation.
        
        Returns:
            List of feature names in topological order (dependencies before dependents)
        
        Raises:
            CircularDependencyError: If circular dependency is detected
            MissingDependencyError: If a dependency is missing
        """
        # Validate first
        is_valid, errors = self.validate()
        if not is_valid:
            # Check for circular dependencies
            cycles = self.detect_cycles()
            if cycles:
                raise CircularDependencyError(cycles[0])
            # Otherwise it's a missing dependency
            raise MissingDependencyError("unknown", "unknown")
        
        # Kahn's algorithm for topological sort
        in_degree = {}
        for feature in self.all_features:
            in_degree[feature] = 0
        
        # Calculate in-degrees
        for feature in self.all_features:
            deps = self.get_dependencies(feature)
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0)  # Ensure dep is in dict
                in_degree[feature] = in_degree.get(feature, 0) + 1
        
        # Find all nodes with in-degree 0
        queue = [f for f in self.all_features if in_degree[f] == 0]
        result = []
        
        while queue:
            # Sort for deterministic ordering when multiple nodes have in-degree 0
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # For each neighbor (features that depend on this one)
            dependents = [f for f in self.all_features if node in self.get_dependencies(f)]
            for neighbor in dependents:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.all_features):
            # There's a cycle (shouldn't reach here due to validation)
            cycles = self.detect_cycles()
            raise CircularDependencyError(cycles[0])
        
        return result
