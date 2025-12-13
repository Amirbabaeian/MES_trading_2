"""
Tests for dependency graph resolution.

Tests:
- Adding features to graph
- Circular dependency detection
- Topological sorting
- Missing dependency validation
"""

import pytest

from src.features.core.dependency import DependencyGraph
from src.features.core.errors import CircularDependencyError


# ============================================================================
# Tests: Basic Graph Operations
# ============================================================================

class TestDependencyGraphBasics:
    """Tests for basic graph operations."""
    
    def test_add_feature_no_dependencies(self):
        """Test adding feature with no dependencies."""
        graph = DependencyGraph()
        graph.add_feature("feature_a", [])
        
        assert "feature_a" in graph.all_features
        assert graph.get_dependencies("feature_a") == set()
    
    def test_add_feature_with_dependencies(self):
        """Test adding feature with dependencies."""
        graph = DependencyGraph()
        graph.add_feature("feature_a", [])
        graph.add_feature("feature_b", ["feature_a"])
        
        assert "feature_a" in graph.all_features
        assert "feature_b" in graph.all_features
        assert graph.get_dependencies("feature_b") == {"feature_a"}
    
    def test_get_all_dependencies_direct(self):
        """Test getting direct dependencies."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        
        deps = graph.get_all_dependencies("b")
        assert deps == {"a"}
    
    def test_get_all_dependencies_transitive(self):
        """Test getting transitive dependencies."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["b"])
        
        deps = graph.get_all_dependencies("c")
        assert deps == {"a", "b"}


# ============================================================================
# Tests: Circular Dependency Detection
# ============================================================================

class TestCircularDependencies:
    """Tests for circular dependency detection."""
    
    def test_self_dependency(self):
        """Test detection of self-dependency."""
        graph = DependencyGraph()
        graph.add_feature("a", ["a"])
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
    
    def test_simple_cycle(self):
        """Test detection of simple 2-node cycle."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["a"])
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
    
    def test_three_node_cycle(self):
        """Test detection of 3-node cycle."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["c"])
        graph.add_feature("c", ["a"])
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
    
    def test_no_cycle_linear_chain(self):
        """Test that linear chain has no cycles."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["b"])
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
    
    def test_no_cycle_multiple_roots(self):
        """Test DAG with multiple roots has no cycles."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", [])
        graph.add_feature("c", ["a", "b"])
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0


# ============================================================================
# Tests: Validation
# ============================================================================

class TestValidation:
    """Tests for graph validation."""
    
    def test_validate_valid_graph(self):
        """Test validation of valid graph."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["a", "b"])
        
        is_valid, errors = graph.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_circular_graph(self):
        """Test validation detects cycles."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["a"])
        
        is_valid, errors = graph.validate()
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_missing_dependency(self):
        """Test validation detects missing dependencies."""
        graph = DependencyGraph()
        graph.add_feature("a", ["nonexistent"])
        
        is_valid, errors = graph.validate()
        assert not is_valid
        assert len(errors) > 0


# ============================================================================
# Tests: Topological Sort
# ============================================================================

class TestTopologicalSort:
    """Tests for topological sorting."""
    
    def test_topological_sort_no_dependencies(self):
        """Test topological sort with no dependencies."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", [])
        graph.add_feature("c", [])
        
        order = graph.topological_sort()
        
        # All must be present
        assert len(order) == 3
        assert set(order) == {"a", "b", "c"}
    
    def test_topological_sort_linear_chain(self):
        """Test topological sort with linear chain."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["b"])
        
        order = graph.topological_sort()
        
        # a must come before b, b must come before c
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
    
    def test_topological_sort_dag(self):
        """Test topological sort with DAG."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", [])
        graph.add_feature("c", ["a", "b"])
        graph.add_feature("d", ["c"])
        
        order = graph.topological_sort()
        
        # Verify dependencies are satisfied
        for i, feature in enumerate(order):
            for dep in graph.get_dependencies(feature):
                assert order.index(dep) < i
    
    def test_topological_sort_circular_raises(self):
        """Test that circular dependency raises error."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["a"])
        
        with pytest.raises(CircularDependencyError):
            graph.topological_sort()
    
    def test_topological_sort_deterministic(self):
        """Test that topological sort is deterministic."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", [])
        graph.add_feature("c", [])
        
        order1 = graph.topological_sort()
        order2 = graph.topological_sort()
        
        assert order1 == order2
    
    def test_topological_sort_complex_dag(self):
        """Test topological sort with complex DAG."""
        graph = DependencyGraph()
        # Create a diamond dependency pattern
        # a -> b -> d
        # a -> c -> d
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["a"])
        graph.add_feature("d", ["b", "c"])
        
        order = graph.topological_sort()
        
        # Verify: a before b,c and b,c before d
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")
