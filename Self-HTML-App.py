
import streamlit as st
import re
from bs4 import BeautifulSoup, Comment
from typing import Dict, List, Tuple, Optional
import html

# Configure page
st.set_page_config(
    page_title="Semantic HTML Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            return {'error': f"HTML parsing error: {str(e)}"}
        
        analysis = {
            'heading_hierarchy': self._analyze_heading_hierarchy(soup),
            'semantic_usage': self._analyze_semantic_usage(soup),
            'structural_elements': self._analyze_structural_elements(soup),
            'accessibility': self._analyze_accessibility(soup),
            'form_semantics': self._analyze_form_semantics(soup),
            'list_usage': self._analyze_list_usage(soup),
            'text_semantics': self._analyze_text_semantics(soup),
            'total_elements': len(soup.find_all()),
            'issues': [],
            'suggestions': []
        }
        
        # Calculate overall score
        analysis['score'] = self._calculate_score(analysis)
        
        return analysis
    
    def _analyze_heading_hierarchy(self, soup: BeautifulSoup) -> Dict:
        """Analyze heading hierarchy (h1-h6)."""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        hierarchy = {}
        issues = []
        
        if not headings:
            issues.append("No headings found")
            return {'headings': [], 'hierarchy': hierarchy, 'issues': issues, 'score': 0}
        
        # Check for h1
        h1_count = len(soup.find_all('h1'))
        if h1_count == 0:
            issues.append("Missing h1 element")
        elif h1_count > 1:
            issues.append(f"Multiple h1 elements found ({h1_count})")
        
        # Analyze hierarchy
        previous_level = 0
        for heading in headings:
            level = int(heading.name[1])
            hierarchy[heading.name] = hierarchy.get(heading.name, 0) + 1
            
            if level > previous_level + 1:
                issues.append(f"Heading hierarchy skip: {heading.name} after h{previous_level}")
            
            previous_level = level
        
        score = max(0, 15 - len(issues) * 3)
        return {
            'headings': [{'tag': h.name, 'text': h.get_text().strip()[:50]} for h in headings],
            'hierarchy': hierarchy,
            'issues': issues,
            'score': score
        }
    
    def _analyze_semantic_usage(self, soup: BeautifulSoup) -> Dict:
        """Analyze semantic vs non-semantic element usage."""
        all_elements = soup.find_all()
        divs = soup.find_all('div')
        spans = soup.find_all('span')
        
        semantic_count = 0
        for category in self.semantic_elements.values():
            for element in category:
                semantic_count += len(soup.find_all(element))
        
        non_semantic_count = len(divs) + len(spans)
        total_content_elements = len(all_elements) - len(soup.find_all(['html', 'head', 'body', 'title', 'meta', 'link', 'style', 'script']))
        
        if total_content_elements == 0:
            semantic_ratio = 0
        else:
            semantic_ratio = semantic_count / total_content_elements
        
        # Score based on semantic ratio
        score = min(20, int(semantic_ratio * 20))
        
        return {
            'semantic_count': semantic_count,
            'non_semantic_count': non_semantic_count,
            'semantic_ratio': semantic_ratio,
            'divs': len(divs),
            'spans': len(spans),
            'score': score
        }
    
    def _analyze_structural_elements(self, soup: BeautifulSoup) -> Dict:
        """Analyze presence of structural semantic elements."""
        found_elements = {}
        missing_elements = []
        
        for element in self.semantic_elements['structural']:
            found = soup.find_all(element)
            found_elements[element] = len(found)
            if len(found) == 0:
                missing_elements.append(element)
        
        # Score based on structural elements present
        present_count = sum(1 for count in found_elements.values() if count > 0)
        score = min(20, int((present_count / len(self.semantic_elements['structural'])) * 20))
        
        return {
            'found_elements': found_elements,
            'missing_elements': missing_elements,
            'score': score
        }
    
    def _analyze_accessibility(self, soup: BeautifulSoup) -> Dict:
        """Analyze accessibility features."""
        issues = []
        score = 15  # Start with full points
        
        # Check images for alt attributes
        images = soup.find_all('img')
        images_without_alt = [img for img in images if not img.get('alt')]
        if images_without_alt:
            issues.append(f"{len(images_without_alt)} images missing alt attributes")
            score -= min(5, len(images_without_alt))
        
        # Check form labels
        inputs = soup.find_all('input')
        inputs_without_labels = []
        for inp in inputs:
            if inp.get('type') not in ['hidden', 'submit', 'button']:
                input_id = inp.get('id')
                if not input_id or not soup.find('label', {'for': input_id}):
                    inputs_without_labels.append(inp)
        
        if inputs_without_labels:
            issues.append(f"{len(inputs_without_labels)} form inputs missing labels")
            score -= min(5, len(inputs_without_labels))
        
        # Check for ARIA attributes
        aria_elements = soup.find_all(attrs={"aria-label": True}) + soup.find_all(attrs={"aria-describedby": True})
        aria_count = len(aria_elements)
        
        # Check for lang attribute
        html_tag = soup.find('html')
        if not html_tag or not html_tag.get('lang'):
            issues.append("Missing lang attribute on html element")
            score -= 2
        
        return {
            'images_total': len(images),
            'images_without_alt': len(images_without_alt),
            'inputs_without_labels': len(inputs_without_labels),
            'aria_elements': aria_count,
            'issues': issues,
            'score': max(0, score)
        }
    
    def _analyze_form_semantics(self, soup: BeautifulSoup) -> Dict:
        """Analyze form semantic structure."""
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
            inputs_in_form = form.find_all(['input', 'select', 'textarea'])
            if len(inputs_in_form) > 5 and not form.find('fieldset'):
                issues.append("Complex form without fieldset grouping")
                score -= 3
        
        return {
            'forms_count': len(forms),
            'fieldsets': len(fieldsets),
            'legends': len(legends),
            'labels': len(labels),
            'issues': issues,
            'score': max(0, score)
        }
    
    def _analyze_list_usage(self, soup: BeautifulSoup) -> Dict:
        """Analyze proper list usage."""
        lists = soup.find_all(['ul', 'ol', 'dl'])
        score = 10
        
        # Check for potential list items that aren't in lists
        potential_lists = 0
        divs = soup.find_all('div')
        for div in divs:
            children = div.find_all(recursive=False)
            if len(children) > 2 and all(child.name == 'div' for child in children):
                potential_lists += 1
        
        if potential_lists > 0:
            score -= min(5, potential_lists * 2)
        
        return {
            'lists_count': len(lists),
            'potential_unlisted_items': potential_lists,
            'score': max(0, score)
        }
    
    def _analyze_text_semantics(self, soup: BeautifulSoup) -> Dict:
        """Analyze text-level semantic usage."""
        semantic_text_elements = 0
        for element in self.semantic_elements['text_semantic']:
            semantic_text_elements += len(soup.find_all(element))
        
        # Check for potential semantic opportunities
        bold_tags = len(soup.find_all('b'))
        italic_tags = len(soup.find_all('i'))
        
        score = min(10, semantic_text_elements)
        
        return {
            'semantic_text_count': semantic_text_elements,
            'bold_tags': bold_tags,
            'italic_tags': italic_tags,
            'score': score
        }
    
    def _calculate_score(self, analysis: Dict) -> int:
        """Calculate overall semantic score."""
        total_score = (
            analysis['heading_hierarchy']['score'] +
            analysis['semantic_usage']['score'] +
            analysis['structural_elements']['score'] +
            analysis['accessibility']['score'] +
            analysis['form_semantics']['score'] +
            analysis['list_usage']['score'] +
            analysis['text_semantics']['score']
        )
        return min(100, total_score)
    
    def suggest_improvements(self, html_content: str, analysis: Dict) -> str:
        """Generate improved HTML with semantic suggestions."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except:
            return "Error parsing HTML for improvements"
        
        improvements = []
        
        # Add missing structural elements
        if not soup.find('main'):
            improvements.append("• Add <main> element to wrap primary content")
        
        if not soup.find('header'):
            improvements.append("• Add <header> element for site/page header")
        
        if not soup.find('nav'):
            improvements.append("• Add <nav> element for navigation links")
        
        # Heading improvements
        if analysis['heading_hierarchy']['issues']:
            improvements.append("• Fix heading hierarchy issues:")
            for issue in analysis['heading_hierarchy']['issues']:
                improvements.append(f"  - {issue}")
        
        # Accessibility improvements
        if analysis['accessibility']['images_without_alt'] > 0:
            improvements.append(f"• Add alt attributes to {analysis['accessibility']['images_without_alt']} images")
        
        if analysis['accessibility']['inputs_without_labels'] > 0:
            improvements.append(f"• Add labels to {analysis['accessibility']['inputs_without_labels']} form inputs")
        
        # Semantic element suggestions
        divs = soup.find_all('div')
        if len(divs) > 5:
            improvements.append("• Consider replacing generic <div> elements with semantic alternatives:")
            improvements.append("  - Use <article> for standalone content")
            improvements.append("  - Use <section> for thematic content groups")
            improvements.append("  - Use <aside> for sidebar content")
        
        return "\n".join(improvements) if improvements else "No major improvements needed!"

def get_example_snippets() -> Dict[str, str]:
    """Return example HTML snippets for testing."""
    return {
        "Poor Semantic Structure": '''<div>
    <div>My Website</div>
    <div>
        <div>Home</div>
        <div>About</div>
        <div>Contact</div>
    </div>
    <div>
        <div>Welcome to My Site</div>
        <div>This is some content about our company.</div>
        <div>
            <div>Our Services</div>
            <div>We offer web development services.</div>
        </div>
    </div>
    <div>Copyright 2024</div>
</div>''',
        
        "Good Semantic Structure": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>My Website</title>
</head>
<body>
    <header>
        <h1>My Website</h1>
        <nav>
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
            </header>
            <p>This is some content about our company.</p>
            
            <section>
                <h2>Our Services</h2>
                <p>We offer <strong>professional</strong> web development services.</p>
            </section>
        </article>
    </main>
    
    <footer>
        <p>&copy; <time datetime="2024">2024</time> My Website</p>
    </footer>
</body>
</html>''',
        
        "Form Example": '''<form>
    <fieldset>
        <legend>Contact Information</legend>
        <label for="name">Full Name:</label>
        <input type="text" id="name" name="name" required>
        
        <label for="email">Email Address:</label>
        <input type="email" id="email" name="email" required>
        
        <label for="message">Message:</label>
        <textarea id="message" name="message" rows="4"></textarea>
    </fieldset>
    
    <button type="submit">Send Message</button>
</form>''',
        
        "Accessibility Issues": '''<div>
    <img src="logo.png">
    <h3>Welcome</h3>
    <h1>Main Content</h1>
    <form>
        Name: <input type="text" name="name">
        Email: <input type="email" name="email">
        <input type="submit" value="Submit">
    </form>
</div>'''
    }

def main():
    st.title("🔍 Semantic HTML Analyzer")
    st.markdown("**Analyze and improve your HTML semantic structure for better accessibility and SEO**")
    
    # Initialize analyzer
    analyzer = SemanticHTMLAnalyzer()
    
    # Sidebar with examples
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
        
        st.markdown("---")
        st.markdown("### Scoring Breakdown")
        st.markdown("""
        - **Heading Hierarchy** (15 pts)
        - **Semantic Elements** (20 pts)
        - **Structure** (20 pts)
        - **Accessibility** (15 pts)
        - **Forms** (10 pts)
        - **Lists** (10 pts)
        - **Text Semantics** (10 pts)
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
            if score >= 80:
                score_color = "normal"
                emoji = "🟢"
            elif score >= 60:
                score_color = "normal"
                emoji = "🟡"
            else:
                score_color = "inverse"
                emoji = "🔴"
            
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
