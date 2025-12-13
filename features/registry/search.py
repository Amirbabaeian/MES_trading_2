"""
Feature Search and Discovery Interface.

This module provides powerful search and filtering capabilities for discovering
features in the registry based on various criteria.
"""

from typing import List, Optional, Set, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re

from .catalog import FeatureRegistry, FeatureMetadata, FeatureStatus, get_registry


class SearchField(Enum):
    """Fields that can be searched."""
    NAME = "name"
    DESCRIPTION = "description"
    COMPUTATION_LOGIC = "computation_logic"
    CATEGORY = "category"
    TAGS = "tags"
    AUTHOR = "author"
    ALL = "all"


class SortBy(Enum):
    """Sorting options for search results."""
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    CATEGORY = "category"
    USAGE_COUNT = "usage_count"


@dataclass
class SearchQuery:
    """
    Structured search query for finding features.
    
    Attributes:
        keyword: Search keyword (can be regex pattern)
        fields: Fields to search in (defaults to all)
        category: Filter by category
        assets: Filter by assets (features must support all listed assets)
        tags: Filter by tags (features must have all listed tags)
        status: Filter by feature status
        author: Filter by author
        version: Filter by specific version
        min_date: Filter features created after this date
        max_date: Filter features created before this date
        exclude_deprecated: Exclude deprecated features (default True)
        case_sensitive: Whether search is case-sensitive
        use_regex: Whether keyword is a regex pattern
    """
    keyword: Optional[str] = None
    fields: List[SearchField] = None
    category: Optional[str] = None
    assets: List[str] = None
    tags: List[str] = None
    status: Optional[FeatureStatus] = None
    author: Optional[str] = None
    version: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    exclude_deprecated: bool = True
    case_sensitive: bool = False
    use_regex: bool = False
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = [SearchField.ALL]
        if self.assets is None:
            self.assets = []
        if self.tags is None:
            self.tags = []


class FeatureSearcher:
    """
    Advanced search interface for the feature registry.
    
    Provides methods for searching, filtering, and discovering features
    based on various criteria.
    """
    
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        """
        Initialize the searcher.
        
        Args:
            registry: Feature registry to search (uses global if not provided)
        """
        self.registry = registry or get_registry()
    
    def search(
        self,
        keyword: Optional[str] = None,
        fields: Optional[List[SearchField]] = None,
        **filters
    ) -> List[FeatureMetadata]:
        """
        Search for features matching keyword and filters.
        
        Args:
            keyword: Search term
            fields: Fields to search in
            **filters: Additional filter criteria (category, assets, tags, etc.)
            
        Returns:
            List of matching FeatureMetadata objects
        """
        # Build search query
        query = SearchQuery(
            keyword=keyword,
            fields=fields or [SearchField.ALL],
            **filters
        )
        
        return self.execute_query(query)
    
    def execute_query(self, query: SearchQuery) -> List[FeatureMetadata]:
        """
        Execute a structured search query.
        
        Args:
            query: SearchQuery object with search criteria
            
        Returns:
            List of matching FeatureMetadata objects
        """
        # Start with all features
        features = self.registry.list_all(include_deprecated=not query.exclude_deprecated)
        
        # Apply keyword filter
        if query.keyword:
            features = self._filter_by_keyword(features, query)
        
        # Apply status filter
        if query.status:
            features = [f for f in features if f.status == query.status]
        
        # Apply category filter
        if query.category:
            features = [f for f in features if f.category == query.category]
        
        # Apply assets filter (feature must support all specified assets)
        if query.assets:
            features = [
                f for f in features
                if all(asset in f.assets for asset in query.assets)
            ]
        
        # Apply tags filter (feature must have all specified tags)
        if query.tags:
            features = [
                f for f in features
                if all(tag in f.tags for tag in query.tags)
            ]
        
        # Apply author filter
        if query.author:
            author_pattern = query.author if query.case_sensitive else query.author.lower()
            features = [
                f for f in features
                if f.author and (
                    f.author == author_pattern if query.case_sensitive
                    else f.author.lower() == author_pattern
                )
            ]
        
        # Apply version filter
        if query.version:
            features = [f for f in features if f.current_version == query.version]
        
        # Apply date filters
        if query.min_date:
            features = [f for f in features if f.created_at >= query.min_date]
        if query.max_date:
            features = [f for f in features if f.created_at <= query.max_date]
        
        return features
    
    def _filter_by_keyword(
        self,
        features: List[FeatureMetadata],
        query: SearchQuery
    ) -> List[FeatureMetadata]:
        """Filter features by keyword search."""
        keyword = query.keyword
        if not query.case_sensitive:
            keyword = keyword.lower()
        
        # Compile regex if needed
        pattern = None
        if query.use_regex:
            flags = 0 if query.case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(keyword, flags)
            except re.error:
                # Invalid regex, fall back to literal search
                pattern = None
        
        matching_features = []
        
        for feature in features:
            if self._matches_keyword(feature, keyword, query.fields, pattern, query.case_sensitive):
                matching_features.append(feature)
        
        return matching_features
    
    def _matches_keyword(
        self,
        feature: FeatureMetadata,
        keyword: str,
        fields: List[SearchField],
        pattern: Optional[re.Pattern],
        case_sensitive: bool
    ) -> bool:
        """Check if a feature matches the keyword in specified fields."""
        # Determine which fields to search
        search_all = SearchField.ALL in fields
        
        search_fields = {
            SearchField.NAME: feature.name,
            SearchField.DESCRIPTION: feature.description,
            SearchField.COMPUTATION_LOGIC: feature.computation_logic,
            SearchField.CATEGORY: feature.category or "",
            SearchField.TAGS: " ".join(feature.tags),
            SearchField.AUTHOR: feature.author or ""
        }
        
        for field, value in search_fields.items():
            if not search_all and field not in fields:
                continue
            
            search_value = value if case_sensitive else value.lower()
            
            if pattern:
                if pattern.search(search_value):
                    return True
            else:
                if keyword in search_value:
                    return True
        
        return False
    
    def search_by_keyword(self, keyword: str, case_sensitive: bool = False) -> List[FeatureMetadata]:
        """
        Simple keyword search across all fields.
        
        Args:
            keyword: Search term
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching features
        """
        return self.search(keyword=keyword, case_sensitive=case_sensitive)
    
    def find_by_category(self, category: str) -> List[FeatureMetadata]:
        """
        Find all features in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of features in the category
        """
        return self.registry.list_by_category(category)
    
    def find_by_asset(self, asset: str) -> List[FeatureMetadata]:
        """
        Find all features applicable to a specific asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            List of features supporting the asset
        """
        return self.registry.list_by_asset(asset)
    
    def find_by_tag(self, tag: str) -> List[FeatureMetadata]:
        """
        Find all features with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of features with the tag
        """
        return self.registry.list_by_tag(tag)
    
    def find_by_tags(self, tags: List[str], match_all: bool = True) -> List[FeatureMetadata]:
        """
        Find features with specific tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, feature must have all tags; if False, any tag
            
        Returns:
            List of matching features
        """
        if not tags:
            return []
        
        if match_all:
            # Feature must have all tags
            features = self.registry.list_all()
            return [f for f in features if all(tag in f.tags for tag in tags)]
        else:
            # Feature must have at least one tag
            feature_names = set()
            for tag in tags:
                feature_names.update(f.name for f in self.registry.list_by_tag(tag))
            return [self.registry.get(name) for name in feature_names if self.registry.get(name)]
    
    def find_similar(
        self,
        feature_name: str,
        similarity_threshold: float = 0.3
    ) -> List[FeatureMetadata]:
        """
        Find features similar to a given feature.
        
        Similarity is based on shared category, tags, and assets.
        
        Args:
            feature_name: Name of the reference feature
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar features, sorted by similarity
        """
        reference = self.registry.get(feature_name)
        if not reference:
            return []
        
        all_features = self.registry.list_all()
        similar_features = []
        
        for feature in all_features:
            if feature.name == feature_name:
                continue
            
            score = self._calculate_similarity(reference, feature)
            if score >= similarity_threshold:
                similar_features.append((feature, score))
        
        # Sort by similarity score (descending)
        similar_features.sort(key=lambda x: x[1], reverse=True)
        
        return [f for f, _ in similar_features]
    
    def _calculate_similarity(
        self,
        feature1: FeatureMetadata,
        feature2: FeatureMetadata
    ) -> float:
        """Calculate similarity score between two features."""
        score = 0.0
        weights = {
            'category': 0.3,
            'tags': 0.4,
            'assets': 0.2,
            'author': 0.1
        }
        
        # Category match
        if feature1.category and feature2.category and feature1.category == feature2.category:
            score += weights['category']
        
        # Tag overlap
        if feature1.tags and feature2.tags:
            common_tags = feature1.tags & feature2.tags
            tag_similarity = len(common_tags) / max(len(feature1.tags), len(feature2.tags))
            score += weights['tags'] * tag_similarity
        
        # Asset overlap
        if feature1.assets and feature2.assets:
            common_assets = set(feature1.assets) & set(feature2.assets)
            asset_similarity = len(common_assets) / max(len(feature1.assets), len(feature2.assets))
            score += weights['assets'] * asset_similarity
        
        # Same author
        if feature1.author and feature2.author and feature1.author == feature2.author:
            score += weights['author']
        
        return score
    
    def filter_by_criteria(
        self,
        features: Optional[List[FeatureMetadata]] = None,
        predicate: Callable[[FeatureMetadata], bool] = None,
        **kwargs
    ) -> List[FeatureMetadata]:
        """
        Filter features using a custom predicate function.
        
        Args:
            features: List of features to filter (uses all if not provided)
            predicate: Custom filter function
            **kwargs: Named filters (category, status, etc.)
            
        Returns:
            Filtered list of features
        """
        if features is None:
            features = self.registry.list_all()
        
        # Apply keyword filters
        for key, value in kwargs.items():
            if hasattr(FeatureMetadata, key):
                features = [f for f in features if getattr(f, key) == value]
        
        # Apply custom predicate
        if predicate:
            features = [f for f in features if predicate(f)]
        
        return features
    
    def sort_results(
        self,
        features: List[FeatureMetadata],
        sort_by: SortBy = SortBy.NAME,
        reverse: bool = False
    ) -> List[FeatureMetadata]:
        """
        Sort feature results.
        
        Args:
            features: List of features to sort
            sort_by: Field to sort by
            reverse: Sort in reverse order
            
        Returns:
            Sorted list of features
        """
        if sort_by == SortBy.NAME:
            return sorted(features, key=lambda f: f.name, reverse=reverse)
        elif sort_by == SortBy.CREATED_AT:
            return sorted(features, key=lambda f: f.created_at, reverse=reverse)
        elif sort_by == SortBy.UPDATED_AT:
            return sorted(features, key=lambda f: f.updated_at, reverse=reverse)
        elif sort_by == SortBy.CATEGORY:
            return sorted(features, key=lambda f: f.category or "", reverse=reverse)
        else:
            return features
    
    def get_recommendations(
        self,
        category: Optional[str] = None,
        asset: Optional[str] = None,
        limit: int = 10
    ) -> List[FeatureMetadata]:
        """
        Get recommended features based on criteria.
        
        Args:
            category: Filter by category
            asset: Filter by asset
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended features
        """
        features = self.registry.list_all()
        
        if category:
            features = [f for f in features if f.category == category]
        
        if asset:
            features = [f for f in features if asset in f.assets]
        
        # Sort by most recently updated
        features = self.sort_results(features, sort_by=SortBy.UPDATED_AT, reverse=True)
        
        return features[:limit]


def search(keyword: str, **filters) -> List[FeatureMetadata]:
    """
    Convenient function for searching features.
    
    Args:
        keyword: Search keyword
        **filters: Additional filter criteria
        
    Returns:
        List of matching features
    """
    searcher = FeatureSearcher()
    return searcher.search(keyword=keyword, **filters)


def find_by_category(category: str) -> List[FeatureMetadata]:
    """Find features by category."""
    searcher = FeatureSearcher()
    return searcher.find_by_category(category)


def find_by_asset(asset: str) -> List[FeatureMetadata]:
    """Find features by asset."""
    searcher = FeatureSearcher()
    return searcher.find_by_asset(asset)


def find_by_tag(tag: str) -> List[FeatureMetadata]:
    """Find features by tag."""
    searcher = FeatureSearcher()
    return searcher.find_by_tag(tag)
