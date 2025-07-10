# Project Portfolio Assistant

A sophisticated AI-powered chatbot application built with Chainlit and LangChain that helps users manage and discuss their project portfolios. The assistant can analyze projects, provide insights, and help with portfolio optimization.

## Features

- **AI-Powered Chat Interface**: Built with Chainlit for a modern, responsive chat experience
- **Project Portfolio Management**: Upload, analyze, and discuss your projects
- **Intelligent Agent System**: Uses LangChain agents for sophisticated reasoning and task execution
- **Memory Persistence**: Maintains conversation context across sessions
- **Docker Support**: Easy deployment with containerization
- **Modern UI**: Clean, professional interface for portfolio discussions

## Tech Stack

- **Frontend**: Chainlit (Python web framework)
- **AI/ML**: LangChain, Anthropic Claude
- **Database**: SQLite (for conversation memory)
- **Containerization**: Docker
- **Language**: Python 3.11+

## Installation

### Prerequisites

- Python 3.11 or higher
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/project_portfolio_assistant.git
   cd project_portfolio_assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

5. **Run the application**
   ```bash
   chainlit run chainlit_app.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t project-portfolio-assistant .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_api_key project-portfolio-assistant
   ```

## Usage

1. Start the application using one of the methods above
2. Open your browser and navigate to `http://localhost:8000`
3. Begin chatting with the AI assistant about your project portfolio
4. Upload project files or documents for analysis
5. Ask questions about portfolio optimization, project analysis, or general advice

## Project Structure

```
project_portfolio_assistant/
├── chainlit_app.py          # Main application entry point
├── enhanced_agent_manager.py # Core agent management system
├── chatbot/                 # Chatbot-related modules
│   └── langchain_agent.py   # LangChain agent implementation
├── images/                  # UI images and assets
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── chainlit.md             # Chainlit configuration
└── README.md               # This file
```

## Configuration

The application can be configured through:

- **Environment Variables**: API keys and configuration settings
- **Chainlit Configuration**: UI settings in `chainlit.md`
- **Agent Configuration**: Customize agent behavior in `enhanced_agent_manager.py`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Check the documentation in the code comments
- Review the Chainlit and LangChain documentation

## Acknowledgments

- Built with [Chainlit](https://chainlit.io/)
- Powered by [LangChain](https://langchain.com/)
- AI capabilities provided by [Anthropic Claude](https://www.anthropic.com/) 
