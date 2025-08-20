"""
Prompt Templates
================
Pre-defined prompt templates for common tasks.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


async def analysis_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Data analysis prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Analysis prompt text
    """
    try:
        base_prompt = """Data Analysis Framework
=======================

Objective: Perform comprehensive data analysis

Steps to Follow:
1. Data Understanding
   - Examine data structure and types
   - Identify key variables and relationships
   - Check for data quality issues

2. Exploratory Analysis
   - Calculate summary statistics
   - Identify patterns and trends
   - Detect outliers and anomalies

3. Statistical Analysis
   - Perform correlation analysis
   - Test hypotheses if applicable
   - Apply appropriate statistical tests

4. Visualization
   - Create relevant charts and graphs
   - Highlight key findings visually
   - Ensure clarity and readability

5. Insights and Recommendations
   - Summarize key findings
   - Provide actionable insights
   - Suggest next steps

"""
        
        if context:
            if "dataset" in context:
                base_prompt += f"\nDataset: {context['dataset']}\n"
            if "focus_areas" in context:
                base_prompt += f"\nFocus Areas: {', '.join(context['focus_areas'])}\n"
            if "constraints" in context:
                base_prompt += f"\nConstraints: {context['constraints']}\n"
        
        base_prompt += f"\nGenerated: {datetime.now().isoformat()}"
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating analysis prompt: {e}")
        return f"Error: {str(e)}"


async def summary_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Summary generation prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Summary prompt text
    """
    try:
        base_prompt = """Summary Generation Guide
========================

Task: Create a comprehensive summary

Requirements:
• Capture main points and key information
• Maintain logical flow and structure
• Use clear and concise language
• Preserve critical details
• Eliminate redundancy

Structure:
1. Executive Summary (2-3 sentences)
2. Key Points (bullet list)
3. Supporting Details (if needed)
4. Conclusions/Implications

Guidelines:
- Keep summary to 20-30% of original length
- Use active voice when possible
- Include relevant metrics and data
- Highlight action items if present

"""
        
        if context:
            if "content_type" in context:
                base_prompt += f"Content Type: {context['content_type']}\n"
            if "target_audience" in context:
                base_prompt += f"Target Audience: {context['target_audience']}\n"
            if "max_length" in context:
                base_prompt += f"Maximum Length: {context['max_length']} words\n"
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating summary prompt: {e}")
        return f"Error: {str(e)}"


async def debug_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Debugging assistance prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Debug prompt text
    """
    try:
        base_prompt = """Debugging Guide
===============

Problem: Debug and resolve issues

Systematic Approach:
1. Issue Identification
   □ Reproduce the problem
   □ Document error messages
   □ Note unexpected behavior
   □ Identify affected components

2. Information Gathering
   □ Check logs and error traces
   □ Review recent changes
   □ Examine system state
   □ Collect relevant metrics

3. Root Cause Analysis
   □ Form hypotheses
   □ Test each hypothesis
   □ Isolate problem area
   □ Identify root cause

4. Solution Development
   □ Design fix approach
   □ Consider side effects
   □ Plan implementation
   □ Prepare rollback strategy

5. Testing and Validation
   □ Test the fix locally
   □ Verify in staging
   □ Check edge cases
   □ Monitor after deployment

Common Issues to Check:
• Configuration errors
• Permission problems
• Resource constraints
• Network connectivity
• Data format mismatches
• Timing/race conditions
• External dependencies

"""
        
        if context:
            if "error_type" in context:
                base_prompt += f"\nError Type: {context['error_type']}\n"
            if "component" in context:
                base_prompt += f"Affected Component: {context['component']}\n"
            if "environment" in context:
                base_prompt += f"Environment: {context['environment']}\n"
            if "stack_trace" in context:
                base_prompt += f"\nStack Trace:\n{context['stack_trace'][:500]}...\n"
        
        base_prompt += "\nDebug Session Started: " + datetime.now().isoformat()
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating debug prompt: {e}")
        return f"Error: {str(e)}"


async def optimization_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Performance optimization prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Optimization prompt text
    """
    try:
        base_prompt = """Performance Optimization Strategy
==================================

Goal: Optimize system/code performance

Optimization Process:
1. Performance Profiling
   - Measure current performance
   - Identify bottlenecks
   - Collect baseline metrics
   - Prioritize problem areas

2. Analysis Phase
   - Time complexity analysis
   - Space complexity review
   - Resource utilization check
   - Dependency assessment

3. Optimization Techniques
   Algorithm Optimization:
   • Use efficient data structures
   • Reduce algorithmic complexity
   • Eliminate unnecessary operations
   • Apply caching strategies
   
   Code Optimization:
   • Remove redundant code
   • Optimize loops and iterations
   • Use built-in functions
   • Minimize I/O operations
   
   System Optimization:
   • Configure resource limits
   • Optimize database queries
   • Implement connection pooling
   • Use async/parallel processing

4. Implementation
   - Apply optimizations incrementally
   - Test after each change
   - Document modifications
   - Monitor impact

5. Validation
   - Compare with baseline
   - Run performance tests
   - Check for regressions
   - Validate improvements

Key Metrics to Track:
• Response time
• Throughput
• CPU usage
• Memory consumption
• I/O operations
• Cache hit ratio

"""
        
        if context:
            if "target_metric" in context:
                base_prompt += f"\nTarget Metric: {context['target_metric']}\n"
            if "current_performance" in context:
                base_prompt += f"Current Performance: {context['current_performance']}\n"
            if "target_improvement" in context:
                base_prompt += f"Target Improvement: {context['target_improvement']}\n"
            if "constraints" in context:
                base_prompt += f"Constraints: {context['constraints']}\n"
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating optimization prompt: {e}")
        return f"Error: {str(e)}"


async def code_review_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Code review prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Code review prompt text
    """
    try:
        base_prompt = """Code Review Checklist
=====================

Objective: Comprehensive code review

Review Categories:

1. Code Quality
   □ Clear and descriptive naming
   □ Consistent coding style
   □ Appropriate comments
   □ No code duplication
   □ SOLID principles followed

2. Functionality
   □ Meets requirements
   □ Handles edge cases
   □ Input validation present
   □ Error handling complete
   □ Expected output produced

3. Performance
   □ Efficient algorithms used
   □ No unnecessary operations
   □ Resource usage optimized
   □ Caching implemented where needed
   □ Database queries optimized

4. Security
   □ Input sanitization
   □ Authentication checks
   □ Authorization verified
   □ No sensitive data exposed
   □ Secure communication used

5. Testing
   □ Unit tests present
   □ Test coverage adequate
   □ Edge cases tested
   □ Integration tests included
   □ Tests are maintainable

6. Documentation
   □ Function/method documentation
   □ Complex logic explained
   □ API documentation updated
   □ README updated if needed
   □ Change log maintained

7. Maintainability
   □ Code is modular
   □ Dependencies minimal
   □ Configuration externalized
   □ Logging appropriate
   □ Monitoring hooks present

Review Feedback Format:
• [CRITICAL]: Must fix before merge
• [MAJOR]: Should fix, impacts quality
• [MINOR]: Consider fixing, nice to have
• [QUESTION]: Clarification needed
• [PRAISE]: Good practice noted

"""
        
        if context:
            if "language" in context:
                base_prompt += f"\nLanguage: {context['language']}\n"
            if "framework" in context:
                base_prompt += f"Framework: {context['framework']}\n"
            if "pr_size" in context:
                base_prompt += f"PR Size: {context['pr_size']} lines\n"
            if "focus_areas" in context:
                base_prompt += f"Focus Areas: {', '.join(context['focus_areas'])}\n"
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating code review prompt: {e}")
        return f"Error: {str(e)}"


async def documentation_prompt(
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Documentation generation prompt template.
    
    Args:
        context: Optional context for the prompt
    
    Returns:
        Documentation prompt text
    """
    try:
        base_prompt = """Documentation Template
======================

Purpose: Create comprehensive documentation

Documentation Structure:

1. Overview
   - Brief description
   - Purpose and goals
   - Target audience
   - Prerequisites

2. Getting Started
   - Installation steps
   - Basic configuration
   - Quick start guide
   - First example

3. Core Concepts
   - Key terminology
   - Architecture overview
   - Main components
   - Data flow

4. Detailed Usage
   - Feature descriptions
   - API reference
   - Configuration options
   - Advanced features

5. Examples
   - Basic examples
   - Common use cases
   - Advanced scenarios
   - Best practices

6. Troubleshooting
   - Common issues
   - Error messages
   - Debug techniques
   - FAQ section

7. Reference
   - API documentation
   - Configuration reference
   - Glossary
   - External resources

Documentation Guidelines:
• Use clear, concise language
• Include code examples
• Add diagrams where helpful
• Keep sections focused
• Use consistent formatting
• Update version information
• Include timestamps

Format Standards:
- Headers: Use proper hierarchy
- Code: Use syntax highlighting
- Lists: Use for steps/options
- Tables: Use for comparisons
- Links: Ensure all are valid
- Images: Include alt text

"""
        
        if context:
            if "doc_type" in context:
                base_prompt += f"\nDocumentation Type: {context['doc_type']}\n"
            if "target_audience" in context:
                base_prompt += f"Target Audience: {context['target_audience']}\n"
            if "scope" in context:
                base_prompt += f"Scope: {context['scope']}\n"
            if "version" in context:
                base_prompt += f"Version: {context['version']}\n"
        
        base_prompt += f"\nGenerated: {datetime.now().isoformat()}"
        
        return base_prompt
        
    except Exception as e:
        logger.error(f"Error generating documentation prompt: {e}")
        return f"Error: {str(e)}"