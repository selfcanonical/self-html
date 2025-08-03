
import streamlit as st
import re
from bs4 import BeautifulSoup, Comment
from typing import Dict, List, Tuple, Optional
import html

# Configure page
st.set_page_config(
    page_title="Self HTML - Semantic HTML Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_html(html_content: str) -> BeautifulSoup:
    """
    Parse HTML content with optimized parser selection and fallbacks.
    Tries lxml first for performance, then html5lib for lenient parsing,
    finally falls back to built-in html.parser.
    """
    if not html_content or not html_content.strip():
        raise ValueError("Empty HTML content provided")
    
    parsers = [
        ('lxml', 'Fast XML/HTML parser'),
        ('html5lib', 'Lenient HTML5 parser'),
        ('html.parser', 'Built-in Python parser')
    ]
    
    last_error = None
    
    for parser_name, description in parsers:
        try:
            soup = BeautifulSoup(html_content, parser_name)
            # Verify the parser actually worked by checking if we got content
            if soup and (soup.find() or soup.get_text().strip()):
                return soup
        except Exception as e:
            last_error = e
            continue
    
    # If all parsers failed, raise the last error
    raise Exception(f"Failed to parse HTML with any available parser. Last error: {last_error}")

class SemanticHTMLAnalyzer:
    def __init__(self):
        self.semantic_elements = {
            'structural': ['header', 'main', 'nav', 'footer', 'article', 'section', 'aside'],
            'text_semantic': ['strong', 'em', 'mark', 'small', 'del', 'ins', 'sub', 'sup', 
                             'code', 'kbd', 'samp', 'var', 'time', 'abbr', 'address', 
                             'cite', 'q', 'blockquote', 'dfn'],
            'list_elements': ['ul', 'ol', 'dl', 'li', 'dt', 'dd'],
            'form_elements': ['form', 'fieldset', 'legend', 'label', 'input', 'textarea', 
                             'select', 'option', 'optgroup', 'button'],
            'media_elements': ['figure', 'figcaption', 'img', 'audio', 'video'],
            'interactive': ['details', 'summary', 'dialog']
        }
        
    def analyze_html(self, html_content: str) -> Dict:
        """Analyze HTML content and return comprehensive results."""
        try:
            soup = parse_html(html_content)
        except Exception as e:
            return {'error': f"HTML parsing error: {str(e)}"}
        
        # Perform comprehensive analysis
        analysis = {
            'heading_hierarchy': self._analyze_heading_hierarchy(soup),
            'semantic_usage': self._analyze_semantic_usage(soup),
            'structural_elements': self._analyze_structural_elements(soup),
            'accessibility': self._analyze_accessibility(soup),
            'form_semantics': self._analyze_form_semantics(soup),
            'list_usage': self._analyze_list_usage(soup),
            'text_semantics': self._analyze_text_semantics(soup),
            'total_elements': len(soup.find_all()) if soup else 0,
            'issues': [],
            'suggestions': []
        }
        
        # Calculate overall score
        analysis['score'] = self._calculate_score(analysis)
        
        return analysis
    
    def _analyze_heading_hierarchy(self, soup: BeautifulSoup) -> Dict:
        """Analyze heading hierarchy (h1-h6) with enhanced error handling."""
        try:
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            hierarchy = {}
            issues = []
            
            if not headings:
                issues.append("No headings found")
                return {'headings': [], 'hierarchy': hierarchy, 'issues': issues, 'score': 0}
            
            # Check for h1
            h1_elements = soup.find_all('h1')
            h1_count = len(h1_elements)
            
            if h1_count == 0:
                issues.append("Missing h1 element - every page should have exactly one h1")
            elif h1_count > 1:
                issues.append(f"Multiple h1 elements found ({h1_count}) - use only one h1 per page")
            
            # Analyze hierarchy and build structure
            previous_level = 0
            for heading in headings:
                try:
                    level = int(heading.name[1])
                    hierarchy[heading.name] = hierarchy.get(heading.name, 0) + 1
                    
                    # Check for hierarchy skips
                    if previous_level > 0 and level > previous_level + 1:
                        issues.append(f"Heading hierarchy skip: {heading.name} used after h{previous_level}")
                    
                    previous_level = level
                except (ValueError, AttributeError, IndexError):
                    issues.append(f"Invalid heading element found: {heading}")
            
            # Calculate score based on hierarchy quality
            base_score = 15
            score = max(0, base_score - len(issues) * 3)
            
            # Extract heading data safely
            heading_data = []
            for h in headings:
                try:
                    text = h.get_text().strip()
                    heading_data.append({
                        'tag': h.name,
                        'text': text[:50] + ('...' if len(text) > 50 else '')
                    })
                except:
                    heading_data.append({'tag': h.name, 'text': '[Unable to extract text]'})
            
            return {
                'headings': heading_data,
                'hierarchy': hierarchy,
                'issues': issues,
                'score': score
            }
            
        except Exception as e:
            return {
                'headings': [],
                'hierarchy': {},
                'issues': [f"Error analyzing headings: {str(e)}"],
                'score': 0
            }
    
    def _analyze_semantic_usage(self, soup: BeautifulSoup) -> Dict:
        """Analyze semantic vs non-semantic element usage with error handling."""
        try:
            all_elements = soup.find_all()
            divs = soup.find_all('div')
            spans = soup.find_all('span')
            
            # Count semantic elements
            semantic_count = 0
            for category in self.semantic_elements.values():
                for element in category:
                    try:
                        semantic_count += len(soup.find_all(element))
                    except:
                        continue
            
            non_semantic_count = len(divs) + len(spans)
            
            # Filter out non-content elements for ratio calculation
            excluded_tags = ['html', 'head', 'body', 'title', 'meta', 'link', 'style', 'script', 'noscript']
            content_elements = [elem for elem in all_elements if elem.name not in excluded_tags]
            total_content_elements = len(content_elements)
            
            if total_content_elements == 0:
                semantic_ratio = 0
            else:
                semantic_ratio = min(1.0, semantic_count / total_content_elements)
            
            # Score based on semantic ratio (0-20 points)
            score = min(20, int(semantic_ratio * 20))
            
            return {
                'semantic_count': semantic_count,
                'non_semantic_count': non_semantic_count,
                'semantic_ratio': semantic_ratio,
                'divs': len(divs),
                'spans': len(spans),
                'total_content_elements': total_content_elements,
                'score': score
            }
            
        except Exception as e:
            return {
                'semantic_count': 0,
                'non_semantic_count': 0,
                'semantic_ratio': 0,
                'divs': 0,
                'spans': 0,
                'total_content_elements': 0,
                'score': 0,
                'error': f"Error analyzing semantic usage: {str(e)}"
            }
    
    def _analyze_structural_elements(self, soup: BeautifulSoup) -> Dict:
        """Analyze presence of structural semantic elements."""
        try:
            found_elements = {}
            missing_elements = []
            
            for element in self.semantic_elements['structural']:
                try:
                    found = soup.find_all(element)
                    count = len(found)
                    found_elements[element] = count
                    
                    if count == 0:
                        missing_elements.append(element)
                except:
                    found_elements[element] = 0
                    missing_elements.append(element)
            
            # Calculate score based on structural elements present
            present_count = sum(1 for count in found_elements.values() if count > 0)
            total_structural = len(self.semantic_elements['structural'])
            score = min(20, int((present_count / total_structural) * 20))
            
            return {
                'found_elements': found_elements,
                'missing_elements': missing_elements,
                'present_count': present_count,
                'total_structural': total_structural,
                'score': score
            }
            
        except Exception as e:
            return {
                'found_elements': {},
                'missing_elements': self.semantic_elements['structural'],
                'present_count': 0,
                'total_structural': len(self.semantic_elements['structural']),
                'score': 0,
                'error': f"Error analyzing structural elements: {str(e)}"
            }
    
    def _analyze_accessibility(self, soup: BeautifulSoup) -> Dict:
        """Analyze accessibility features with comprehensive error handling."""
        try:
            issues = []
            score = 15  # Start with full accessibility points
            
            # Check images for alt attributes
            images = soup.find_all('img')
            images_without_alt = []
            
            for img in images:
                try:
                    alt_attr = img.get('alt')
                    if alt_attr is None:
                        images_without_alt.append(img)
                except:
                    images_without_alt.append(img)
            
            if images_without_alt:
                issue_count = len(images_without_alt)
                issues.append(f"{issue_count} image{'s' if issue_count > 1 else ''} missing alt attributes")
                score -= min(5, issue_count)
            
            # Check form inputs for labels
            inputs = soup.find_all('input')
            inputs_without_labels = []
            
            for inp in inputs:
                try:
                    input_type = inp.get('type', '').lower()
                    if input_type not in ['hidden', 'submit', 'button', 'reset']:
                        input_id = inp.get('id')
                        if input_id:
                            # Look for associated label
                            label = soup.find('label', {'for': input_id})
                            if not label:
                                inputs_without_labels.append(inp)
                        else:
                            # No ID, check if wrapped in label
                            parent_label = inp.find_parent('label')
                            if not parent_label:
                                inputs_without_labels.append(inp)
                except:
                    inputs_without_labels.append(inp)
            
            if inputs_without_labels:
                issue_count = len(inputs_without_labels)
                issues.append(f"{issue_count} form input{'s' if issue_count > 1 else ''} missing proper labels")
                score -= min(5, issue_count)
            
            # Check for ARIA attributes usage
            aria_elements = []
            try:
                aria_elements.extend(soup.find_all(attrs={"aria-label": True}))
                aria_elements.extend(soup.find_all(attrs={"aria-describedby": True}))
                aria_elements.extend(soup.find_all(attrs={"aria-labelledby": True}))
                aria_elements.extend(soup.find_all(attrs={"role": True}))
            except:
                pass
            
            aria_count = len(set(aria_elements))  # Remove duplicates
            
            # Check for lang attribute on html element
            try:
                html_tag = soup.find('html')
                if not html_tag or not html_tag.get('lang'):
                    issues.append("Missing lang attribute on html element")
                    score -= 2
            except:
                issues.append("Could not verify lang attribute")
                score -= 1
            
            # Check for proper table headers
            tables = soup.find_all('table')
            tables_without_th = 0
            for table in tables:
                try:
                    if not table.find('th'):
                        tables_without_th += 1
                except:
                    tables_without_th += 1
            
            if tables_without_th > 0:
                issues.append(f"{tables_without_th} table{'s' if tables_without_th > 1 else ''} without proper headers (th elements)")
                score -= min(2, tables_without_th)
            
            return {
                'images_total': len(images),
                'images_without_alt': len(images_without_alt),
                'inputs_total': len(inputs),
                'inputs_without_labels': len(inputs_without_labels),
                'aria_elements': aria_count,
                'tables_without_th': tables_without_th,
                'issues': issues,
                'score': max(0, score)
            }
            
        except Exception as e:
            return {
                'images_total': 0,
                'images_without_alt': 0,
                'inputs_total': 0,
                'inputs_without_labels': 0,
                'aria_elements': 0,
                'tables_without_th': 0,
                'issues': [f"Error analyzing accessibility: {str(e)}"],
                'score': 0
            }
    
    def _analyze_form_semantics(self, soup: BeautifulSoup) -> Dict:
        """Analyze form semantic structure with error handling."""
        try:
            forms = soup.find_all('form')
            if not forms:
                return {'forms_count': 0, 'score': 10}  # No forms, full points
            
            score = 10
            issues = []
            
            fieldsets = soup.find_all('fieldset')
            legends = soup.find_all('legend')
            labels = soup.find_all('label')
            
            # Check if complex forms use fieldsets
            for form in forms:
                try:
                    inputs_in_form = form.find_all(['input', 'select', 'textarea'])
                    form_fieldsets = form.find_all('fieldset')
                    
                    # If form has many inputs but no fieldsets, suggest grouping
                    if len(inputs_in_form) > 5 and len(form_fieldsets) == 0:
                        issues.append("Complex form without fieldset grouping")
                        score -= 3
                        
                    # Check for orphaned legends (legends without fieldsets)
                    form_legends = form.find_all('legend')
                    for legend in form_legends:
                        if not legend.find_parent('fieldset'):
                            issues.append("Legend element not inside fieldset")
                            score -= 1
                            
                except Exception:
                    continue
            
            return {
                'forms_count': len(forms),
                'fieldsets': len(fieldsets),
                'legends': len(legends),
                'labels': len(labels),
                'issues': issues,
                'score': max(0, score)
            }
            
        except Exception as e:
            return {
                'forms_count': 0,
                'fieldsets': 0,
                'legends': 0,
                'labels': 0,
                'issues': [f"Error analyzing forms: {str(e)}"],
                'score': 0
            }
    
    def _analyze_list_usage(self, soup: BeautifulSoup) -> Dict:
        """Analyze proper list usage with error handling."""
        try:
            lists = soup.find_all(['ul', 'ol', 'dl'])
            score = 10
            potential_lists = 0
            
            # Check for potential list items that aren't in lists
            divs = soup.find_all('div')
            for div in divs:
                try:
                    # Get direct children only
                    children = [child for child in div.children if child.name]
                    
                    # If div has multiple similar child elements, might be a list
                    if len(children) > 2:
                        child_tags = [child.name for child in children]
                        # If most children have the same tag, could be list items
                        if len(set(child_tags)) == 1 and child_tags[0] in ['div', 'p', 'span']:
                            potential_lists += 1
                except:
                    continue
            
            if potential_lists > 0:
                score -= min(5, potential_lists * 2)
            
            # Check for proper list nesting
            nested_issues = 0
            for ul in soup.find_all('ul'):
                try:
                    # Check if nested lists are properly structured
                    nested_lists = ul.find_all(['ul', 'ol'])
                    for nested in nested_lists:
                        if not nested.find_parent('li'):
                            nested_issues += 1
                except:
                    continue
            
            if nested_issues > 0:
                score -= min(2, nested_issues)
            
            return {
                'lists_count': len(lists),
                'potential_unlisted_items': potential_lists,
                'nested_issues': nested_issues,
                'score': max(0, score)
            }
            
        except Exception as e:
            return {
                'lists_count': 0,
                'potential_unlisted_items': 0,
                'nested_issues': 0,
                'score': 0,
                'error': f"Error analyzing lists: {str(e)}"
            }
    
    def _analyze_text_semantics(self, soup: BeautifulSoup) -> Dict:
        """Analyze text-level semantic usage with error handling."""
        try:
            semantic_text_elements = 0
            semantic_breakdown = {}
            
            # Count semantic text elements
            for element in self.semantic_elements['text_semantic']:
                try:
                    count = len(soup.find_all(element))
                    if count > 0:
                        semantic_breakdown[element] = count
                        semantic_text_elements += count
                except:
                    continue
            
            # Check for potential semantic opportunities
            bold_tags = len(soup.find_all('b'))
            italic_tags = len(soup.find_all('i'))
            
            # Score based on semantic text usage
            score = min(10, semantic_text_elements // 2)
            
            # Bonus points for variety of semantic elements
            if len(semantic_breakdown) > 3:
                score += 2
            
            score = min(10, score)
            
            return {
                'semantic_text_count': semantic_text_elements,
                'semantic_breakdown': semantic_breakdown,
                'bold_tags': bold_tags,
                'italic_tags': italic_tags,
                'variety_bonus': len(semantic_breakdown) > 3,
                'score': score
            }
            
        except Exception as e:
            return {
                'semantic_text_count': 0,
                'semantic_breakdown': {},
                'bold_tags': 0,
                'italic_tags': 0,
                'variety_bonus': False,
                'score': 0,
                'error': f"Error analyzing text semantics: {str(e)}"
            }
    
    def _calculate_score(self, analysis: Dict) -> int:
        """Calculate overall semantic score with error handling."""
        try:
            total_score = (
                analysis.get('heading_hierarchy', {}).get('score', 0) +
                analysis.get('semantic_usage', {}).get('score', 0) +
                analysis.get('structural_elements', {}).get('score', 0) +
                analysis.get('accessibility', {}).get('score', 0) +
                analysis.get('form_semantics', {}).get('score', 0) +
                analysis.get('list_usage', {}).get('score', 0) +
                analysis.get('text_semantics', {}).get('score', 0)
            )
            return min(100, max(0, total_score))
        except:
            return 0
    
    def suggest_improvements(self, html_content: str, analysis: Dict) -> str:
        """Generate improved HTML suggestions with error handling."""
        try:
            soup = parse_html(html_content)
        except Exception as e:
            return f"Error parsing HTML for improvements: {str(e)}"
        
        improvements = []
        
        try:
            # Structural improvements
            if not soup.find('main'):
                improvements.append("• Add <main> element to wrap primary page content")
            
            if not soup.find('header'):
                improvements.append("• Add <header> element for site/page header")
            
            if not soup.find('nav') and soup.find('a'):
                improvements.append("• Add <nav> element to wrap navigation links")
            
            if not soup.find('footer'):
                improvements.append("• Add <footer> element for page footer content")
            
            # Heading improvements
            heading_issues = analysis.get('heading_hierarchy', {}).get('issues', [])
            if heading_issues:
                improvements.append("• Fix heading hierarchy issues:")
                for issue in heading_issues[:3]:  # Limit to first 3 issues
                    improvements.append(f"  - {issue}")
            
            # Accessibility improvements
            accessibility = analysis.get('accessibility', {})
            if accessibility.get('images_without_alt', 0) > 0:
                count = accessibility['images_without_alt']
                improvements.append(f"• Add alt attributes to {count} image{'s' if count > 1 else ''}")
            
            if accessibility.get('inputs_without_labels', 0) > 0:
                count = accessibility['inputs_without_labels']
                improvements.append(f"• Add proper labels to {count} form input{'s' if count > 1 else ''}")
            
            # Semantic element suggestions
            semantic_usage = analysis.get('semantic_usage', {})
            div_count = semantic_usage.get('divs', 0)
            
            if div_count > 5:
                improvements.append("• Replace generic <div> elements with semantic alternatives:")
                improvements.append("  - Use <article> for standalone, reusable content")
                improvements.append("  - Use <section> for thematic content groups")
                improvements.append("  - Use <aside> for sidebar or tangential content")
            
            # List improvements
            list_analysis = analysis.get('list_usage', {})
            if list_analysis.get('potential_unlisted_items', 0) > 0:
                improvements.append("• Convert grouped content to proper lists using <ul>, <ol>, or <dl>")
            
            # Text semantic improvements
            text_analysis = analysis.get('text_semantics', {})
            if text_analysis.get('bold_tags', 0) > 0:
                improvements.append("• Replace <b> tags with <strong> for important content")
            
            if text_analysis.get('italic_tags', 0) > 0:
                improvements.append("• Replace <i> tags with <em> for emphasized content")
            
            # Form improvements
            form_issues = analysis.get('form_semantics', {}).get('issues', [])
            if form_issues:
                improvements.append("• Improve form structure:")
                for issue in form_issues:
                    improvements.append(f"  - {issue}")
            
            # If no issues found
            if not improvements:
                return "Excellent! Your HTML structure demonstrates good semantic practices. No major improvements needed."
            
            return "\n".join(improvements)
            
        except Exception as e:
            return f"Error generating improvement suggestions: {str(e)}"

def get_example_snippets() -> Dict[str, str]:
    """Return example HTML snippets for testing."""
    return {
        "Poor Semantic Structure": '''<div class="page">
    <div class="header">My Website</div>
    <div class="menu">
        <div>Home</div>
        <div>About</div>
        <div>Contact</div>
    </div>
    <div class="content">
        <div class="title">Welcome to My Site</div>
        <div>This is some content about our company.</div>
        <div>
            <div class="section-title">Our Services</div>
            <div>We offer web development services.</div>
        </div>
    </div>
    <div class="footer">Copyright 2024</div>
</div>''',
        
        "Good Semantic Structure": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Welcome to My Site - My Website</title>
</head>
<body>
    <header>
        <h1>My Website</h1>
        <nav aria-label="Main navigation">
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <header>
                <h1>Welcome to My Site</h1>
                <p>Published on <time datetime="2024-07-28">July 28, 2024</time></p>
            </header>
            <p>This is some content about our <strong>professional</strong> company.</p>
            
            <section>
                <h2>Our Services</h2>
                <p>We offer <em>comprehensive</em> web development services.</p>
                <ul>
                    <li>Frontend Development</li>
                    <li>Backend Development</li>
                    <li>UI/UX Design</li>
                </ul>
            </section>
        </article>
    </main>
    
    <footer>
        <p>&copy; <time datetime="2024">2024</time> My Website. All rights reserved.</p>
        <address>
            Contact us at <a href="mailto:info@mywebsite.com">info@mywebsite.com</a>
        </address>
    </footer>
</body>
</html>''',
        
        "Form with Good Semantics": '''<form action="/contact" method="post">
    <fieldset>
        <legend>Contact Information</legend>
        
        <label for="fullname">Full Name (required):</label>
        <input type="text" id="fullname" name="fullname" required aria-describedby="name-help">
        <small id="name-help">Enter your first and last name</small>
        
        <label for="email">Email Address (required):</label>
        <input type="email" id="email" name="email" required>
        
        <label for="phone">Phone Number:</label>
        <input type="tel" id="phone" name="phone">
    </fieldset>
    
    <fieldset>
        <legend>Message Details</legend>
        
        <label for="subject">Subject:</label>
        <select id="subject" name="subject">
            <option value="">Choose a topic</option>
            <option value="general">General Inquiry</option>
            <option value="support">Support Request</option>
            <option value="feedback">Feedback</option>
        </select>
        
        <label for="message">Your Message (required):</label>
        <textarea id="message" name="message" rows="5" required></textarea>
    </fieldset>
    
    <button type="submit">Send Message</button>
    <button type="reset">Clear Form</button>
</form>''',
        
        "Accessibility Issues Example": '''<div>
    <img src="company-logo.png">
    <h3>About Us</h3>
    <h1>Welcome</h1>
    <h4>Our Mission</h4>
    
    <form>
        Name: <input type="text" name="name">
        Email: <input type="email" name="email">
        <input type="submit" value="Submit">
    </form>
    
    <table>
        <tr>
            <td>Product</td>
            <td>Price</td>
        </tr>
        <tr>
            <td>Widget</td>
            <td>\$10</td>
        </tr>
    </table>
</div>'''
    }

def display_score_indicator(score: int) -> Tuple[str, str]:
    """Return emoji and color class for score visualization."""
    if score >= 90:
        return "🟢", "success"
    elif score >= 80:
        return "🟢", "success"  
    elif score >= 70:
        return "🟡", "warning"
    elif score >= 60:
        return "🟡", "warning"
    elif score >= 40:
        return "🟠", "warning"
    else:
        return "🔴", "error"

def main():
    st.title("🔍 Semantic HTML Analyzer")
    st.markdown("**Analyze and improve your HTML semantic structure for better accessibility, SEO, and maintainability**")
    
    # Initialize analyzer
    analyzer = SemanticHTMLAnalyzer()
    
    # Sidebar with examples and information
    with st.sidebar:
        st.header("📚 Example Snippets")
        examples = get_example_snippets()
        
        selected_example = st.selectbox(
            "Choose an example to analyze:",
            ["Select an example..."] + list(examples.keys())
        )
        
        if selected_example != "Select an example...":
            if st.button("Load Example", key="load_example"):
                st.session_state.html_input = examples[selected_example]
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Scoring System")
        st.markdown("""
        **Total: 100 Points**
        - **Heading Hierarchy** (15 pts)
          - Proper h1-h6 usage
          - Single h1 per page
          - No hierarchy skips
        - **Semantic Elements** (20 pts)
          - Semantic vs generic ratio
        - **Document Structure** (20 pts)
          - header, main, nav, footer
          - article, section, aside
        - **Accessibility** (15 pts)
          - Alt attributes, labels
          - ARIA attributes, lang
        - **Form Semantics** (10 pts)
          - Fieldsets, legends, labels
        - **List Usage** (10 pts)
          - Proper list structures
        - **Text Semantics** (10 pts)
          - strong, em, time, etc.
        """)
    
    # Main input area
    st.header("📝 HTML Input")
    
    # Initialize session state
    if 'html_input' not in st.session_state:
        st.session_state.html_input = ""
    
    html_input = st.text_area(
        "Enter your HTML content:",
        value=st.session_state.html_input,
        height=300,
        key="html_content"
    )
    
    # Update session state
    st.session_state.html_input = html_input
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze HTML", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
        if clear_button:
            st.session_state.html_input = ""
            st.rerun()
    
    if analyze_button and html_input.strip():
        # Perform analysis
        with st.spinner("Analyzing HTML structure..."):
            analysis = analyzer.analyze_html(html_input)
        
        if 'error' in analysis:
            st.error(analysis['error'])
            return
        
        # Display results
        st.header("📊 Analysis Results")
        
        # Overall score
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            score = analysis['score']
            emoji, score_color = display_score_indicator(score)
            
            st.metric(
                label=f"{emoji} Overall Semantic Score",
                value=f"{score}/100",
                delta=f"{score}% semantic quality"
            )
        
        with col2:
            st.metric("Total Elements", analysis['total_elements'])
        
        with col3:
            semantic_ratio = analysis['semantic_usage']['semantic_ratio']
            st.metric("Semantic Ratio", f"{semantic_ratio:.1%}")
        
        # Progress bar
        st.progress(score / 100)
        
        # Detailed analysis
        st.header("🔍 Detailed Analysis")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs(["Structure & Semantics", "Accessibility", "Heading Analysis", "Improvements"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("🏗️ Structural Elements", expanded=True):
                    struct_analysis = analysis['structural_elements']
                    st.metric("Score", f"{struct_analysis['score']}/20")
                    
                    st.write("**Found Elements:**")
                    for element, count in struct_analysis['found_elements'].items():
                        status = "✅" if count > 0 else "❌"
                        st.write(f"{status} `{element}`: {count}")
                    
                    if struct_analysis['missing_elements']:
                        st.write("**Missing Elements:**")
                        for element in struct_analysis['missing_elements']:
                            st.write(f"• `{element}`")
                
                with st.expander("📝 Text Semantics"):
                    text_analysis = analysis['text_semantics']
                    st.metric("Score", f"{text_analysis['score']}/10")
                    st.write(f"Semantic text elements: {text_analysis['semantic_text_count']}")
                    st.write(f"Generic bold tags: {text_analysis['bold_tags']}")
                    st.write(f"Generic italic tags: {text_analysis['italic_tags']}")
            
            with col2:
                with st.expander("🔖 Semantic Usage", expanded=True):
                    semantic_analysis = analysis['semantic_usage']
                    st.metric("Score", f"{semantic_analysis['score']}/20")
                    
                    st.write(f"**Semantic elements:** {semantic_analysis['semantic_count']}")
                    st.write(f"**Generic divs:** {semantic_analysis['divs']}")
                    st.write(f"**Generic spans:** {semantic_analysis['spans']}")
                    st.write(f"**Semantic ratio:** {semantic_analysis['semantic_ratio']:.1%}")
                
                with st.expander("📋 Lists & Forms"):
                    list_analysis = analysis['list_usage']
                    form_analysis = analysis['form_semantics']
                    
                    st.write("**Lists:**")
                    st.write(f"Score: {list_analysis['score']}/10")
                    st.write(f"Lists found: {list_analysis['lists_count']}")
                    
                    st.write("**Forms:**")
                    st.write(f"Score: {form_analysis['score']}/10")
                    st.write(f"Forms: {form_analysis['forms_count']}")
                    if 'fieldsets' in form_analysis:
                        st.write(f"Fieldsets: {form_analysis['fieldsets']}")
        
        with tab2:
            accessibility = analysis['accessibility']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accessibility Score", f"{accessibility['score']}/15")
                
                st.write("**Image Accessibility:**")
                st.write(f"• Total images: {accessibility['images_total']}")
                st.write(f"• Missing alt text: {accessibility['images_without_alt']}")
                
                st.write("**Form Accessibility:**")
                st.write(f"• Inputs without labels: {accessibility['inputs_without_labels']}")
            
            with col2:
                st.write("**ARIA Usage:**")
                st.write(f"• Elements with ARIA attributes: {accessibility['aria_elements']}")
                
                if accessibility['issues']:
                    st.write("**Issues Found:**")
                    for issue in accessibility['issues']:
                        st.write(f"⚠️ {issue}")
        
        with tab3:
            heading_analysis = analysis['heading_hierarchy']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Heading Score", f"{heading_analysis['score']}/15")
                
                if heading_analysis['headings']:
                    st.write("**Headings Found:**")
                    for heading in heading_analysis['headings']:
                        st.write(f"`{heading['tag']}`: {heading['text']}")
                else:
                    st.write("No headings found")
            
            with col2:
                st.write("**Heading Distribution:**")
                for tag, count in heading_analysis['hierarchy'].items():
                    st.write(f"`{tag}`: {count}")
                
                if heading_analysis['issues']:
                    st.write("**Issues:**")
                    for issue in heading_analysis['issues']:
                        st.write(f"⚠️ {issue}")
        
        with tab4:
            st.subheader("💡 Suggested Improvements")
            
            improvements = analyzer.suggest_improvements(html_input, analysis)
            
            if improvements:
                st.code(improvements, language=None)
            else:
                st.success("Great job! Your HTML structure looks semantically sound.")
            
            # Show before/after comparison if score is low
            if score < 70:
                st.subheader("📋 HTML Structure Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original HTML:**")
                    st.code(html_input, language='html')
                
                with col2:
                    st.write("**Semantic Structure Tips:**")
                    st.markdown("""
                    ```html
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <title>Page Title</title>
                    </head>
                    <body>
                        <header>
                            <h1>Main Title</h1>
                            <nav>
                                <ul>
                                    <li><a href="#">Link</a></li>
                                </ul>
                            </nav>
                        </header>
                        
                        <main>
                            <article>
                                <h2>Section Title</h2>
                                <p>Content...</p>
                            </article>
                        </main>
                        
                        <footer>
                            <p>&copy; 2024</p>
                        </footer>
                    </body>
                    </html>
                    ```
                    """)
    
    elif analyze_button:
        st.warning("Please enter some HTML content to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔍 <strong>Semantic HTML Analyzer</strong> | Built with Streamlit</p>
        <p>Improve your HTML structure for better accessibility, SEO, and maintainability</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
