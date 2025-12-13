"""
Feature Documentation Generator.

This module automatically generates comprehensive documentation from feature
metadata, including markdown and HTML output formats.
"""

from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime
import json

from .catalog import FeatureRegistry, FeatureMetadata, FeatureStatus, get_registry
from .search import FeatureSearcher


class DocumentationFormat:
    """Base class for documentation format generators."""
    
    def generate(self, features: List[FeatureMetadata], **kwargs) -> str:
        """Generate documentation in the specific format."""
        raise NotImplementedError


class MarkdownFormatter(DocumentationFormat):
    """Generate markdown documentation."""
    
    def generate(
        self,
        features: List[FeatureMetadata],
        title: str = "Feature Catalog",
        include_toc: bool = True,
        include_deprecated: bool = False,
        group_by_category: bool = True
    ) -> str:
        """
        Generate markdown documentation.
        
        Args:
            features: List of features to document
            title: Document title
            include_toc: Include table of contents
            include_deprecated: Include deprecated features
            group_by_category: Group features by category
            
        Returns:
            Markdown formatted documentation
        """
        lines = []
        
        # Title and metadata
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append(f"**Total Features:** {len(features)}")
        lines.append("")
        
        # Filter deprecated if needed
        if not include_deprecated:
            features = [f for f in features if f.status != FeatureStatus.DEPRECATED]
        
        # Group by category if requested
        if group_by_category:
            categorized = self._group_by_category(features)
            
            # Table of contents
            if include_toc:
                lines.append("## Table of Contents")
                lines.append("")
                for category in sorted(categorized.keys()):
                    category_name = category or "Uncategorized"
                    lines.append(f"- [{category_name}](#{self._anchor(category_name)})")
                lines.append("")
            
            # Feature sections by category
            for category in sorted(categorized.keys()):
                category_name = category or "Uncategorized"
                category_features = categorized[category]
                lines.append(f"## {category_name}")
                lines.append("")
                
                for feature in sorted(category_features, key=lambda f: f.name):
                    lines.extend(self._format_feature(feature))
                    lines.append("")
        else:
            # Table of contents
            if include_toc:
                lines.append("## Table of Contents")
                lines.append("")
                for feature in sorted(features, key=lambda f: f.name):
                    lines.append(f"- [{feature.name}](#{self._anchor(feature.name)})")
                lines.append("")
            
            # All features
            lines.append("## Features")
            lines.append("")
            for feature in sorted(features, key=lambda f: f.name):
                lines.extend(self._format_feature(feature))
                lines.append("")
        
        return "\n".join(lines)
    
    def _group_by_category(self, features: List[FeatureMetadata]) -> Dict[str, List[FeatureMetadata]]:
        """Group features by category."""
        categorized = {}
        for feature in features:
            category = feature.category or ""
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(feature)
        return categorized
    
    def _format_feature(self, feature: FeatureMetadata) -> List[str]:
        """Format a single feature as markdown."""
        lines = []
        
        # Feature header
        status_badge = self._status_badge(feature.status)
        lines.append(f"### {feature.name} {status_badge}")
        lines.append("")
        
        # Description
        lines.append(f"**Description:** {feature.description}")
        lines.append("")
        
        # Computation logic
        if feature.computation_logic:
            lines.append(f"**Computation:** {feature.computation_logic}")
            lines.append("")
        
        # Metadata table
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Version | `{feature.current_version}` |")
        lines.append(f"| Category | {feature.category or 'N/A'} |")
        lines.append(f"| Author | {feature.author or 'N/A'} |")
        lines.append(f"| Created | {feature.created_at.strftime('%Y-%m-%d')} |")
        lines.append(f"| Updated | {feature.updated_at.strftime('%Y-%m-%d')} |")
        lines.append(f"| Computation Cost | {feature.computation_cost} |")
        lines.append("")
        
        # Assets
        if feature.assets:
            lines.append(f"**Supported Assets:** {', '.join(f'`{a}`' for a in feature.assets)}")
            lines.append("")
        
        # Tags
        if feature.tags:
            tags = ', '.join(f'`{tag}`' for tag in sorted(feature.tags))
            lines.append(f"**Tags:** {tags}")
            lines.append("")
        
        # Parameters
        if feature.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param_name, param_spec in feature.parameters.items():
                if isinstance(param_spec, dict):
                    param_type = param_spec.get('type', 'Any')
                    param_default = param_spec.get('default')
                    param_required = param_spec.get('required', False)
                    req_str = " *(required)*" if param_required else ""
                    default_str = f" = `{param_default}`" if param_default else ""
                    lines.append(f"- `{param_name}`: {param_type}{default_str}{req_str}")
                else:
                    lines.append(f"- `{param_name}`: {param_spec}")
            lines.append("")
        
        # Dependencies
        if feature.data_dependencies:
            lines.append(f"**Dependencies:** {', '.join(f'`{d}`' for d in feature.data_dependencies)}")
            lines.append("")
        
        # Version history
        if feature.version_history and len(feature.version_history) > 1:
            lines.append("**Version History:**")
            lines.append("")
            for version in reversed(feature.version_history):
                lines.append(f"- `{version.version}` ({version.created_at.strftime('%Y-%m-%d')})")
                if version.changes:
                    lines.append(f"  - {version.changes}")
            lines.append("")
        
        # Deprecation info
        if feature.deprecation_info:
            lines.append("⚠️ **Deprecation Notice:**")
            lines.append("")
            lines.append(f"- **Reason:** {feature.deprecation_info.get('reason', 'N/A')}")
            if 'replacement' in feature.deprecation_info:
                lines.append(f"- **Replacement:** `{feature.deprecation_info['replacement']}`")
            if 'deprecated_in' in feature.deprecation_info:
                lines.append(f"- **Deprecated in:** {feature.deprecation_info['deprecated_in']}")
            if 'remove_in' in feature.deprecation_info:
                lines.append(f"- **Will be removed in:** {feature.deprecation_info['remove_in']}")
            lines.append("")
        
        # Example usage
        if feature.example_usage:
            lines.append("**Example Usage:**")
            lines.append("")
            lines.append("```python")
            lines.append(feature.example_usage)
            lines.append("```")
            lines.append("")
        
        lines.append("---")
        
        return lines
    
    def _status_badge(self, status: FeatureStatus) -> str:
        """Generate a status badge."""
        badges = {
            FeatureStatus.ACTIVE: "![Active](https://img.shields.io/badge/status-active-green)",
            FeatureStatus.DEPRECATED: "![Deprecated](https://img.shields.io/badge/status-deprecated-red)",
            FeatureStatus.EXPERIMENTAL: "![Experimental](https://img.shields.io/badge/status-experimental-yellow)",
            FeatureStatus.ARCHIVED: "![Archived](https://img.shields.io/badge/status-archived-gray)"
        }
        return badges.get(status, "")
    
    def _anchor(self, text: str) -> str:
        """Generate anchor link for table of contents."""
        return text.lower().replace(" ", "-").replace("_", "-")


class HTMLFormatter(DocumentationFormat):
    """Generate HTML documentation."""
    
    def generate(
        self,
        features: List[FeatureMetadata],
        title: str = "Feature Catalog",
        include_toc: bool = True,
        include_deprecated: bool = False,
        group_by_category: bool = True,
        css_style: Optional[str] = None
    ) -> str:
        """
        Generate HTML documentation.
        
        Args:
            features: List of features to document
            title: Document title
            include_toc: Include table of contents
            include_deprecated: Include deprecated features
            group_by_category: Group features by category
            css_style: Custom CSS style
            
        Returns:
            HTML formatted documentation
        """
        css = css_style or self._default_css()
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append(f"<title>{title}</title>")
        html.append("<meta charset='utf-8'>")
        html.append(f"<style>{css}</style>")
        html.append("</head>")
        html.append("<body>")
        html.append(f"<h1>{title}</h1>")
        html.append(f"<p class='metadata'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p class='metadata'>Total Features: {len(features)}</p>")
        
        # Filter deprecated if needed
        if not include_deprecated:
            features = [f for f in features if f.status != FeatureStatus.DEPRECATED]
        
        # Group by category if requested
        if group_by_category:
            categorized = self._group_by_category(features)
            
            # Table of contents
            if include_toc:
                html.append("<div class='toc'>")
                html.append("<h2>Table of Contents</h2>")
                html.append("<ul>")
                for category in sorted(categorized.keys()):
                    category_name = category or "Uncategorized"
                    anchor = self._anchor(category_name)
                    html.append(f"<li><a href='#{anchor}'>{category_name}</a></li>")
                html.append("</ul>")
                html.append("</div>")
            
            # Feature sections by category
            for category in sorted(categorized.keys()):
                category_name = category or "Uncategorized"
                category_features = categorized[category]
                anchor = self._anchor(category_name)
                html.append(f"<h2 id='{anchor}'>{category_name}</h2>")
                
                for feature in sorted(category_features, key=lambda f: f.name):
                    html.append(self._format_feature_html(feature))
        else:
            # Table of contents
            if include_toc:
                html.append("<div class='toc'>")
                html.append("<h2>Table of Contents</h2>")
                html.append("<ul>")
                for feature in sorted(features, key=lambda f: f.name):
                    anchor = self._anchor(feature.name)
                    html.append(f"<li><a href='#{anchor}'>{feature.name}</a></li>")
                html.append("</ul>")
                html.append("</div>")
            
            # All features
            html.append("<h2>Features</h2>")
            for feature in sorted(features, key=lambda f: f.name):
                html.append(self._format_feature_html(feature))
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _group_by_category(self, features: List[FeatureMetadata]) -> Dict[str, List[FeatureMetadata]]:
        """Group features by category."""
        categorized = {}
        for feature in features:
            category = feature.category or ""
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(feature)
        return categorized
    
    def _format_feature_html(self, feature: FeatureMetadata) -> str:
        """Format a single feature as HTML."""
        anchor = self._anchor(feature.name)
        status_class = feature.status.value
        
        html = [f"<div class='feature feature-{status_class}' id='{anchor}'>"]
        html.append(f"<h3>{feature.name} <span class='badge badge-{status_class}'>{feature.status.value}</span></h3>")
        html.append(f"<p class='description'>{feature.description}</p>")
        
        if feature.computation_logic:
            html.append(f"<p><strong>Computation:</strong> {feature.computation_logic}</p>")
        
        # Metadata table
        html.append("<table class='metadata-table'>")
        html.append("<tr><th>Property</th><th>Value</th></tr>")
        html.append(f"<tr><td>Version</td><td><code>{feature.current_version}</code></td></tr>")
        html.append(f"<tr><td>Category</td><td>{feature.category or 'N/A'}</td></tr>")
        html.append(f"<tr><td>Author</td><td>{feature.author or 'N/A'}</td></tr>")
        html.append(f"<tr><td>Created</td><td>{feature.created_at.strftime('%Y-%m-%d')}</td></tr>")
        html.append(f"<tr><td>Updated</td><td>{feature.updated_at.strftime('%Y-%m-%d')}</td></tr>")
        html.append(f"<tr><td>Computation Cost</td><td>{feature.computation_cost}</td></tr>")
        html.append("</table>")
        
        # Assets, tags, parameters, etc.
        if feature.assets:
            assets_html = ', '.join(f'<code>{a}</code>' for a in feature.assets)
            html.append(f"<p><strong>Supported Assets:</strong> {assets_html}</p>")
        
        if feature.tags:
            tags_html = ', '.join(f'<span class="tag">{tag}</span>' for tag in sorted(feature.tags))
            html.append(f"<p><strong>Tags:</strong> {tags_html}</p>")
        
        if feature.parameters:
            html.append("<p><strong>Parameters:</strong></p>")
            html.append("<ul>")
            for param_name, param_spec in feature.parameters.items():
                if isinstance(param_spec, dict):
                    param_type = param_spec.get('type', 'Any')
                    param_default = param_spec.get('default')
                    default_str = f" = <code>{param_default}</code>" if param_default else ""
                    html.append(f"<li><code>{param_name}</code>: {param_type}{default_str}</li>")
                else:
                    html.append(f"<li><code>{param_name}</code>: {param_spec}</li>")
            html.append("</ul>")
        
        if feature.data_dependencies:
            deps_html = ', '.join(f'<code>{d}</code>' for d in feature.data_dependencies)
            html.append(f"<p><strong>Dependencies:</strong> {deps_html}</p>")
        
        if feature.deprecation_info:
            html.append("<div class='deprecation-warning'>")
            html.append("<strong>⚠️ Deprecation Notice:</strong>")
            html.append(f"<p>Reason: {feature.deprecation_info.get('reason', 'N/A')}</p>")
            if 'replacement' in feature.deprecation_info:
                html.append(f"<p>Replacement: <code>{feature.deprecation_info['replacement']}</code></p>")
            html.append("</div>")
        
        if feature.example_usage:
            html.append("<p><strong>Example Usage:</strong></p>")
            html.append(f"<pre><code>{feature.example_usage}</code></pre>")
        
        html.append("</div>")
        
        return "\n".join(html)
    
    def _anchor(self, text: str) -> str:
        """Generate anchor link."""
        return text.lower().replace(" ", "-").replace("_", "-")
    
    def _default_css(self) -> str:
        """Return default CSS styling."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 { color: #333; }
        h1 { border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { border-bottom: 2px solid #6c757d; padding-bottom: 8px; margin-top: 30px; }
        .metadata { color: #666; font-size: 0.9em; }
        .toc { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .toc ul { list-style-type: none; padding-left: 0; }
        .toc li { margin: 8px 0; }
        .toc a { color: #007bff; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        .feature {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
        }
        .feature-deprecated { border-left-color: #dc3545; }
        .feature-experimental { border-left-color: #ffc107; }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-active { background: #28a745; color: white; }
        .badge-deprecated { background: #dc3545; color: white; }
        .badge-experimental { background: #ffc107; color: black; }
        .badge-archived { background: #6c757d; color: white; }
        .description { font-size: 1.1em; color: #555; margin: 10px 0; }
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .metadata-table th, .metadata-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .metadata-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        pre code { background: none; padding: 0; }
        .tag {
            display: inline-block;
            background: #e9ecef;
            padding: 2px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 0.9em;
        }
        .deprecation-warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        """


class JSONFormatter(DocumentationFormat):
    """Generate JSON documentation."""
    
    def generate(
        self,
        features: List[FeatureMetadata],
        include_deprecated: bool = False,
        pretty_print: bool = True
    ) -> str:
        """
        Generate JSON documentation.
        
        Args:
            features: List of features to document
            include_deprecated: Include deprecated features
            pretty_print: Pretty print JSON output
            
        Returns:
            JSON formatted documentation
        """
        if not include_deprecated:
            features = [f for f in features if f.status != FeatureStatus.DEPRECATED]
        
        data = {
            'generated_at': datetime.now().isoformat(),
            'total_features': len(features),
            'features': [self._serialize_feature(f) for f in features]
        }
        
        indent = 2 if pretty_print else None
        return json.dumps(data, indent=indent, default=str)
    
    def _serialize_feature(self, feature: FeatureMetadata) -> Dict[str, Any]:
        """Serialize a feature to a dictionary."""
        return feature.to_dict()


class DocumentationGenerator:
    """
    Main documentation generator class.
    
    Handles generation of comprehensive documentation from the feature registry
    in multiple formats (Markdown, HTML, JSON).
    """
    
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        """
        Initialize the documentation generator.
        
        Args:
            registry: Feature registry (uses global if not provided)
        """
        self.registry = registry or get_registry()
        self.formatters = {
            'markdown': MarkdownFormatter(),
            'html': HTMLFormatter(),
            'json': JSONFormatter()
        }
    
    def generate(
        self,
        output_path: Optional[str] = None,
        format: str = 'markdown',
        features: Optional[List[FeatureMetadata]] = None,
        **kwargs
    ) -> str:
        """
        Generate documentation for features.
        
        Args:
            output_path: Optional path to save documentation
            format: Output format ('markdown', 'html', 'json')
            features: List of features (uses all from registry if not provided)
            **kwargs: Additional format-specific options
            
        Returns:
            Generated documentation as string
        """
        if features is None:
            features = self.registry.list_all(include_deprecated=kwargs.get('include_deprecated', False))
        
        formatter = self.formatters.get(format)
        if not formatter:
            raise ValueError(f"Unsupported format: {format}")
        
        documentation = formatter.generate(features, **kwargs)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(documentation)
        
        return documentation
    
    def generate_catalog(
        self,
        output_dir: str,
        formats: List[str] = None,
        **kwargs
    ):
        """
        Generate complete feature catalog in multiple formats.
        
        Args:
            output_dir: Directory to save documentation files
            formats: List of formats to generate (defaults to all)
            **kwargs: Additional options
        """
        if formats is None:
            formats = ['markdown', 'html', 'json']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        features = self.registry.list_all(include_deprecated=kwargs.get('include_deprecated', False))
        
        for fmt in formats:
            ext = {'markdown': 'md', 'html': 'html', 'json': 'json'}[fmt]
            file_path = output_path / f"feature_catalog.{ext}"
            self.generate(
                output_path=str(file_path),
                format=fmt,
                features=features,
                **kwargs
            )
    
    def generate_category_docs(
        self,
        output_dir: str,
        format: str = 'markdown'
    ):
        """
        Generate separate documentation for each category.
        
        Args:
            output_dir: Directory to save documentation files
            format: Output format
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        categories = self.registry.list_categories()
        
        for category in categories:
            features = self.registry.list_by_category(category)
            category_name = category or "uncategorized"
            safe_name = category_name.lower().replace(" ", "_")
            
            ext = {'markdown': 'md', 'html': 'html', 'json': 'json'}[format]
            file_path = output_path / f"{safe_name}.{ext}"
            
            self.generate(
                output_path=str(file_path),
                format=format,
                features=features,
                title=f"{category_name} Features"
            )
    
    def generate_summary(self) -> str:
        """Generate a summary report of the registry."""
        stats = self.registry.get_statistics()
        
        lines = [
            "# Feature Registry Summary",
            "",
            f"**Total Features:** {stats['total_features']}",
            f"**Active Features:** {stats['active_features']}",
            f"**Deprecated Features:** {stats['deprecated_features']}",
            f"**Experimental Features:** {stats['experimental_features']}",
            f"**Categories:** {stats['categories']}",
            f"**Assets:** {stats['assets']}",
            f"**Tags:** {stats['tags']}",
            "",
            "## Status Breakdown",
            ""
        ]
        
        for status, count in stats['status_breakdown'].items():
            lines.append(f"- {status}: {count}")
        
        lines.append("")
        lines.append("## Categories")
        lines.append("")
        
        for category in self.registry.list_categories():
            count = len(self.registry.list_by_category(category))
            lines.append(f"- {category or 'Uncategorized'}: {count} features")
        
        return "\n".join(lines)


def generate_documentation(
    output_path: str,
    format: str = 'markdown',
    **kwargs
) -> str:
    """
    Convenient function to generate feature documentation.
    
    Args:
        output_path: Path to save documentation
        format: Output format
        **kwargs: Additional options
        
    Returns:
        Generated documentation string
    """
    generator = DocumentationGenerator()
    return generator.generate(output_path=output_path, format=format, **kwargs)
