"""Single source of truth for skills and career roles.

Used by the synthetic data generator (now), and by the skill extractor, skill-gap
analysis and job matcher (later modules).
"""

# canonical skill -> (category, [aliases])
SKILLS = {
    # Programming languages
    "python": ("Programming", ["py"]),
    "java": ("Programming", []),
    "javascript": ("Programming", ["js", "ecmascript"]),
    "typescript": ("Programming", ["ts"]),
    "c++": ("Programming", ["cpp"]),
    "c#": ("Programming", ["csharp"]),
    "sql": ("Programming", ["structured query language"]),
    "r": ("Programming", ["r programming"]),
    "kotlin": ("Programming", []),
    "swift": ("Programming", []),
    "bash": ("Programming", ["shell scripting"]),
    # Web frontend
    "html": ("Frontend", ["html5"]),
    "css": ("Frontend", ["css3"]),
    "react": ("Frontend", ["reactjs", "react.js"]),
    "angular": ("Frontend", ["angularjs"]),
    "vue": ("Frontend", ["vuejs", "vue.js"]),
    "redux": ("Frontend", []),
    "tailwind css": ("Frontend", ["tailwind"]),
    "bootstrap": ("Frontend", []),
    "next.js": ("Frontend", ["nextjs"]),
    "cypress": ("Testing", []),
    # Backend
    "node.js": ("Backend", ["nodejs", "node"]),
    "express": ("Backend", ["expressjs", "express.js"]),
    "django": ("Backend", []),
    "flask": ("Backend", []),
    "fastapi": ("Backend", []),
    "spring boot": ("Backend", ["springboot"]),
    "rest api": ("Backend", ["restful api", "rest apis", "restful apis"]),
    "graphql": ("Backend", []),
    "microservices": ("Backend", ["microservice architecture"]),
    # Databases
    "mysql": ("Database", []),
    "postgresql": ("Database", ["postgres"]),
    "mongodb": ("Database", ["mongo"]),
    "sqlite": ("Database", []),
    "redis": ("Database", []),
    "firebase": ("Database", []),
    # Data science / ML
    "machine learning": ("Data/ML", ["ml"]),
    "deep learning": ("Data/ML", ["dl"]),
    "nlp": ("Data/ML", ["natural language processing"]),
    "computer vision": ("Data/ML", ["cv"]),
    "pandas": ("Data/ML", []),
    "numpy": ("Data/ML", []),
    "scikit-learn": ("Data/ML", ["sklearn", "scikit learn"]),
    "tensorflow": ("Data/ML", ["tf"]),
    "pytorch": ("Data/ML", ["torch"]),
    "keras": ("Data/ML", []),
    "spacy": ("Data/ML", []),
    "statistics": ("Data/ML", ["statistical analysis"]),
    "feature engineering": ("Data/ML", []),
    "matplotlib": ("Data/ML", []),
    "seaborn": ("Data/ML", []),
    "a/b testing": ("Data/ML", ["ab testing"]),
    # Analytics / BI
    "excel": ("Analytics", ["microsoft excel", "ms excel"]),
    "power bi": ("Analytics", ["powerbi"]),
    "tableau": ("Analytics", []),
    "data visualization": ("Analytics", ["data visualisation"]),
    "data cleaning": ("Analytics", ["data wrangling"]),
    "etl": ("Analytics", ["etl pipelines"]),
    # DevOps / cloud
    "docker": ("DevOps", []),
    "kubernetes": ("DevOps", ["k8s"]),
    "aws": ("DevOps", ["amazon web services"]),
    "azure": ("DevOps", ["microsoft azure"]),
    "gcp": ("DevOps", ["google cloud"]),
    "jenkins": ("DevOps", []),
    "ci/cd": ("DevOps", ["cicd", "ci cd"]),
    "terraform": ("DevOps", []),
    "ansible": ("DevOps", []),
    "linux": ("DevOps", ["ubuntu"]),
    "git": ("DevOps", ["github", "version control"]),
    "github actions": ("DevOps", []),
    "prometheus": ("DevOps", []),
    "nginx": ("DevOps", []),
    # Mobile
    "android": ("Mobile", ["android development"]),
    "flutter": ("Mobile", []),
    "react native": ("Mobile", []),
    "ios": ("Mobile", []),
    "jetpack compose": ("Mobile", []),
    # Security
    "network security": ("Security", []),
    "penetration testing": ("Security", ["pentesting", "pen testing"]),
    "siem": ("Security", []),
    "cryptography": ("Security", []),
    "owasp": ("Security", ["owasp top 10"]),
    "wireshark": ("Security", []),
    "burp suite": ("Security", []),
    "incident response": ("Security", []),
    "vulnerability assessment": ("Security", []),
    "nmap": ("Security", []),
    # QA
    "selenium": ("Testing", []),
    "junit": ("Testing", []),
    "pytest": ("Testing", []),
    "test automation": ("Testing", ["automated testing"]),
    "manual testing": ("Testing", []),
    "postman": ("Testing", []),
    "jira": ("Testing", []),
}

SOFT_SKILLS = ["communication", "teamwork", "problem solving", "leadership", "agile", "time management"]

DISPLAY_OVERRIDES = {
    "sql": "SQL", "html": "HTML", "css": "CSS", "aws": "AWS", "gcp": "GCP", "nlp": "NLP",
    "ci/cd": "CI/CD", "etl": "ETL", "siem": "SIEM", "owasp": "OWASP", "ios": "iOS",
    "rest api": "REST API", "a/b testing": "A/B Testing", "power bi": "Power BI",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "mongodb": "MongoDB",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "sqlite": "SQLite", "fastapi": "FastAPI",
    "graphql": "GraphQL", "javascript": "JavaScript", "typescript": "TypeScript",
    "spacy": "spaCy", "junit": "JUnit", "github actions": "GitHub Actions",
    "scikit-learn": "scikit-learn", "tailwind css": "Tailwind CSS", "numpy": "NumPy",
}


def display_name(skill: str) -> str:
    """Human-friendly skill spelling for generated resume text."""
    if skill in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[skill]
    return " ".join(w.capitalize() for w in skill.split(" "))


# role -> description, core skills (must-have), secondary skills (nice-to-have),
# typical project nouns (used only by the synthetic generator)
ROLES = {
    "Data Scientist": {
        "description": "Builds statistical and ML models to extract insight and predictions from data.",
        "core": ["python", "machine learning", "statistics", "pandas", "numpy", "scikit-learn", "sql", "data visualization"],
        "secondary": ["deep learning", "tensorflow", "pytorch", "matplotlib", "feature engineering", "nlp", "r", "a/b testing"],
        "nouns": ["a customer churn prediction model", "a sales forecasting pipeline", "an exploratory analysis of retail data", "a fraud detection classifier"],
    },
    "Machine Learning Engineer": {
        "description": "Productionises ML models: training pipelines, serving, and monitoring.",
        "core": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "docker", "git"],
        "secondary": ["keras", "nlp", "computer vision", "aws", "fastapi", "flask", "linux", "ci/cd", "spacy", "numpy"],
        "nouns": ["a model serving pipeline", "an image classification service", "a recommendation engine", "a training workflow for deep models"],
    },
    "Data Analyst": {
        "description": "Turns raw data into reports and dashboards that guide business decisions.",
        "core": ["sql", "excel", "power bi", "tableau", "data visualization", "data cleaning", "statistics"],
        "secondary": ["python", "pandas", "mysql", "etl", "a/b testing", "matplotlib", "seaborn", "postgresql"],
        "nouns": ["an executive sales dashboard", "a monthly KPI report", "a data cleaning workflow", "a customer segmentation analysis"],
    },
    "Frontend Developer": {
        "description": "Builds responsive, accessible user interfaces for web applications.",
        "core": ["html", "css", "javascript", "react", "git", "rest api"],
        "secondary": ["typescript", "redux", "tailwind css", "bootstrap", "next.js", "angular", "vue", "cypress"],
        "nouns": ["a responsive e-commerce interface", "a reusable component library", "a single page dashboard", "a marketing landing page"],
    },
    "Backend Developer": {
        "description": "Designs APIs, services, and databases that power applications.",
        "core": ["rest api", "sql", "git", "microservices", "postgresql"],
        "secondary": ["node.js", "express", "django", "flask", "fastapi", "spring boot", "java", "python", "mongodb", "redis", "docker", "graphql", "mysql"],
        "nouns": ["a payments API", "an order management service", "an authentication service", "a scalable inventory backend"],
    },
    "Full Stack Developer": {
        "description": "Works across frontend, backend, and database layers of web products.",
        "core": ["html", "css", "javascript", "react", "node.js", "express", "mongodb", "rest api", "git"],
        "secondary": ["typescript", "sql", "postgresql", "redux", "docker", "next.js", "aws", "tailwind css", "mysql"],
        "nouns": ["a full stack job portal", "an e-learning platform", "a social media web app", "a booking application"],
    },
    "DevOps Engineer": {
        "description": "Automates build, deployment, and infrastructure for reliable releases.",
        "core": ["docker", "kubernetes", "ci/cd", "linux", "aws", "git", "jenkins"],
        "secondary": ["terraform", "ansible", "azure", "gcp", "prometheus", "nginx", "bash", "github actions", "python"],
        "nouns": ["a CI/CD pipeline", "a containerised deployment", "an infrastructure automation setup", "a monitoring stack"],
    },
    "Mobile App Developer": {
        "description": "Builds native and cross-platform mobile applications.",
        "core": ["android", "kotlin", "java", "rest api", "git", "firebase"],
        "secondary": ["flutter", "react native", "jetpack compose", "ios", "swift", "sqlite"],
        "nouns": ["a fitness tracking app", "a food delivery app", "a chat application", "an expense tracker app"],
    },
    "Cybersecurity Analyst": {
        "description": "Finds and fixes vulnerabilities and responds to security incidents.",
        "core": ["network security", "penetration testing", "linux", "owasp", "vulnerability assessment", "wireshark"],
        "secondary": ["siem", "cryptography", "burp suite", "incident response", "python", "bash", "nmap"],
        "nouns": ["a vulnerability assessment report", "a network monitoring setup", "a web application security audit", "a phishing awareness exercise"],
    },
    "QA / Test Engineer": {
        "description": "Ensures software quality through manual and automated testing.",
        "core": ["manual testing", "test automation", "selenium", "jira", "postman"],
        "secondary": ["pytest", "junit", "cypress", "java", "python", "sql", "git", "ci/cd"],
        "nouns": ["an automated regression suite", "a test plan for a web portal", "an API test collection", "a bug tracking workflow"],
    },
}

# Roles with realistic skill overlap -> used to create "hybrid" resumes (makes the ML task non-trivial).
NEIGHBOURS = {
    "Data Scientist": ["Machine Learning Engineer", "Data Analyst"],
    "Machine Learning Engineer": ["Data Scientist", "DevOps Engineer"],
    "Data Analyst": ["Data Scientist"],
    "Frontend Developer": ["Full Stack Developer"],
    "Backend Developer": ["Full Stack Developer", "DevOps Engineer"],
    "Full Stack Developer": ["Frontend Developer", "Backend Developer"],
    "DevOps Engineer": ["Backend Developer", "Cybersecurity Analyst"],
    "Mobile App Developer": ["Full Stack Developer"],
    "Cybersecurity Analyst": ["DevOps Engineer"],
    "QA / Test Engineer": ["Backend Developer"],
}
GENERIC_NOUNS = ["an internal tool", "a college team project", "a client deliverable", "a proof of concept"]
