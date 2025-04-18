# AI Assistant for NVDA

## Overview
AI Assistant is an NVDA add-on that provides easy access to multiple AI language models directly within your screen reader. With a simple keyboard shortcut, you can send selected text to various AI models and receive instant responses.

## Features
- Support for multiple AI providers:
  - OpenAI (GPT-4 models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Cohere (Command models)
- Configurable settings for each provider
- Customizable response parameters (length, creativity)
- Accessible response display with clean formatting
- Option to have responses read aloud

## Installation
1. Download the latest release from the NVDA add-ons GitHub repository
2. Open the downloaded file and follow the installation prompts
3. Restart NVDA when prompted

## Usage
1. Select text you want to analyze or process with AI
2. Press `Ctrl+Shift+K` to send the selection to your configured AI model
3. A response window will appear with the AI's response
4. Press `Escape` to close the response window

## Configuration
Access the add-on settings through NVDA menu > Preferences > Settings > AI Assistant:

1. **Select AI Service**: Choose between OpenAI, Anthropic, Google, or Cohere
2. **API Key**: Enter your API key for the selected service
3. **Model**: Select the specific model to use
4. **Maximum Tokens**: Set the maximum length of responses
5. **Temperature**: Adjust the creativity level of the AI (0.0-1.0)
6. **Read Response**: Toggle whether responses should be read aloud

## API Keys
You'll need to obtain API keys from the service providers you wish to use:
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://www.anthropic.com/api)
- [Google AI Studio](https://makersuite.google.com/)
- [Cohere API](https://cohere.com/api)

## Requirements
- NVDA 2021.1 or later
- Internet connection
- Valid API key for at least one of the supported services

## Privacy & Security
- Your API keys are stored locally in NVDA's configuration
- Text selections are only sent to the AI provider you've configured
- No data is stored by the add-on beyond the current session

## Troubleshooting
- **No response**: Check your internet connection and API key
- **Error messages**: Verify your API key is valid and has sufficient credits
- **Slow responses**: Large text selections or complex queries may take longer to process

## Author
**Asamoah Emmanuel**  
Email: emmanuelasamoah179@gmail.com

## License
This add-on is distributed under the terms of the GNU General Public License, version 2 or later.

## Acknowledgements
Thanks to the NVDA community for their feedback and support during development.