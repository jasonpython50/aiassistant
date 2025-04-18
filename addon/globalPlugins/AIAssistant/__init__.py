import os
import sys
import threading
import json
import urllib.request
import urllib.error
import base64
from typing import Optional, Dict, Any
import re

import globalPluginHandler
import api
import ui
import wx
import gui
import addonHandler
import scriptHandler
import speech
import config
from gui import settingsDialogs
from gui import guiHelper
from logHandler import log

# Add third-party module path
module_path = os.path.dirname(__file__)
sys.path.insert(0, module_path)

# Configuration specification
SPEC = {
    'openaiApiKey': 'string(default="")',
    'anthropicApiKey': 'string(default="")',
    'geminiApiKey': 'string(default="")',
    'cohereApiKey': 'string(default="")',
    'apiService': 'string(default="openai")',  # Options: openai, anthropic, gemini, cohere
    'selectedModel': 'string(default="")',
    'maxTokens': 'integer(default=500, min=50, max=4000)',
    'temperature': 'float(default=0.7, min=0.0, max=1.0)',
    'read_response': 'boolean(default=True)',
}

# Model options by service
MODEL_OPTIONS = {
    "openai": ["gpt-4-turbo", "gpt-4-omni", "gpt-4"],
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview"],
    "cohere": ["command-a-03-2025"]
}

def clean_markdown(text):
    """Remove common markdown formatting from text."""
    # Replace headers (### Title) with plain text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Replace bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic *text*
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold __text__
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic _text_
    
    # Clean up bullet points
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)  # * Item
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)   # - Item
    
    # Clean up numbered lists (1. Item)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove code blocks and inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    return text

# Text Window Class
class TextWindow(wx.Frame):
    """A simple text window to display AI responses."""

    def __init__(self, text, title, readOnly=True, insertionPoint=0):
        super(TextWindow, self).__init__(wx.GetApp().TopWindow, title=title)
        sizer = wx.BoxSizer(wx.VERTICAL)
        style = wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH
        self.outputCtrl = wx.TextCtrl(self, style=style)
        self.outputCtrl.Bind(wx.EVT_KEY_DOWN, self.onOutputKeyDown)
        sizer.Add(self.outputCtrl, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.outputCtrl.SetValue(text)
        self.outputCtrl.SetFocus()
        self.outputCtrl.SetInsertionPoint(insertionPoint)
        self.Raise()
        self.Maximize()
        self.Show()

    def onOutputKeyDown(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.Close()
        event.Skip()

class AiAssistantSettingsPanel(settingsDialogs.SettingsPanel):
    title = "AI Assistant"
    
    def makeSettings(self, settingsSizer):
        helper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
        
        # API Service selector
        apiServiceChoices = [
            "OpenAI (GPT-4)",
            "Anthropic (Claude)",
            "Google (Gemini)",
            "Cohere (Command)"
        ]
        
        self.apiServiceChoice = helper.addLabeledControl(
            "AI Service:",
            wx.Choice,
            choices=apiServiceChoices
        )
        
        # Set the current selection
        apiService = config.conf["aiAssistant"]["apiService"]
        if apiService == "openai":
            self.apiServiceChoice.SetSelection(0)
        elif apiService == "anthropic":
            self.apiServiceChoice.SetSelection(1)
        elif apiService == "gemini":
            self.apiServiceChoice.SetSelection(2)
        elif apiService == "cohere":
            self.apiServiceChoice.SetSelection(3)
        else:
            self.apiServiceChoice.SetSelection(0)
        
        # Bind the event for changing service
        self.apiServiceChoice.Bind(wx.EVT_CHOICE, self.onApiServiceChange)
        
        # API Key fields
        # OpenAI
        self.openaiApiKeySizer = wx.BoxSizer(wx.VERTICAL)
        self.openaiApiKeyLabel = wx.StaticText(self, label="OpenAI API Key:")
        self.openaiApiKeyEdit = wx.TextCtrl(
            self, 
            value=config.conf["aiAssistant"]["openaiApiKey"],
            style=wx.TE_PASSWORD
        )
        self.openaiApiKeySizer.Add(self.openaiApiKeyLabel, flag=wx.BOTTOM, border=2)
        self.openaiApiKeySizer.Add(self.openaiApiKeyEdit, flag=wx.EXPAND)
        helper.sizer.Add(self.openaiApiKeySizer, flag=wx.EXPAND|wx.BOTTOM, border=10)
        
        # Anthropic
        self.anthropicApiKeySizer = wx.BoxSizer(wx.VERTICAL)
        self.anthropicApiKeyLabel = wx.StaticText(self, label="Anthropic API Key:")
        self.anthropicApiKeyEdit = wx.TextCtrl(
            self, 
            value=config.conf["aiAssistant"]["anthropicApiKey"],
            style=wx.TE_PASSWORD
        )
        self.anthropicApiKeySizer.Add(self.anthropicApiKeyLabel, flag=wx.BOTTOM, border=2)
        self.anthropicApiKeySizer.Add(self.anthropicApiKeyEdit, flag=wx.EXPAND)
        helper.sizer.Add(self.anthropicApiKeySizer, flag=wx.EXPAND|wx.BOTTOM, border=10)
        
        # Google
        self.geminiApiKeySizer = wx.BoxSizer(wx.VERTICAL)
        self.geminiApiKeyLabel = wx.StaticText(self, label="Google Gemini API Key:")
        self.geminiApiKeyEdit = wx.TextCtrl(
            self, 
            value=config.conf["aiAssistant"]["geminiApiKey"],
            style=wx.TE_PASSWORD
        )
        self.geminiApiKeySizer.Add(self.geminiApiKeyLabel, flag=wx.BOTTOM, border=2)
        self.geminiApiKeySizer.Add(self.geminiApiKeyEdit, flag=wx.EXPAND)
        helper.sizer.Add(self.geminiApiKeySizer, flag=wx.EXPAND|wx.BOTTOM, border=10)
        
        # Cohere
        self.cohereApiKeySizer = wx.BoxSizer(wx.VERTICAL)
        self.cohereApiKeyLabel = wx.StaticText(self, label="Cohere API Key:")
        self.cohereApiKeyEdit = wx.TextCtrl(
            self, 
            value=config.conf["aiAssistant"]["cohereApiKey"],
            style=wx.TE_PASSWORD
        )
        self.cohereApiKeySizer.Add(self.cohereApiKeyLabel, flag=wx.BOTTOM, border=2)
        self.cohereApiKeySizer.Add(self.cohereApiKeyEdit, flag=wx.EXPAND)
        helper.sizer.Add(self.cohereApiKeySizer, flag=wx.EXPAND|wx.BOTTOM, border=10)
        
        # Update visible API key fields
        self.updateApiKeyVisibility()
        
        # Model Selection
        self.modelChoices = []
        self.updateModelChoices()
        
        self.modelChoice = helper.addLabeledControl(
            "Model:",
            wx.Choice,
            choices=self.modelChoices
        )
        
        # Set current model selection
        self.updateModelSelection()
        
        # Generation parameters
        self.maxTokensEdit = helper.addLabeledControl(
            "Maximum tokens (response length):",
            wx.SpinCtrl,
            min=50,
            max=4000,
            initial=config.conf["aiAssistant"]["maxTokens"]
        )
        
        self.temperatureSlider = helper.addLabeledControl(
            "Temperature (creativity level):",
            wx.Slider,
            value=int(config.conf["aiAssistant"]["temperature"] * 100),
            minValue=0,
            maxValue=100,
            style=wx.SL_HORIZONTAL | wx.SL_LABELS
        )
        
        # Speech output checkbox
        self.speechCheckbox = helper.addItem(
            wx.CheckBox(self, label="Read response aloud")
        )
        self.speechCheckbox.SetValue(config.conf["aiAssistant"]["read_response"])
        
        # Add description of current model
        descriptionTitle = wx.StaticText(self, label="Model Description")
        self.modelDescription = wx.StaticText(self, style=wx.ST_NO_AUTORESIZE)
        self.modelDescription.Wrap(self.GetSize().width-20)
        
        descSizer = wx.BoxSizer(wx.VERTICAL)
        descSizer.Add(descriptionTitle, flag=wx.BOTTOM, border=5)
        descSizer.Add(self.modelDescription, flag=wx.EXPAND)
        helper.sizer.Add(descSizer, flag=wx.EXPAND|wx.TOP, border=10)
        
        # Update description text
        self.updateModelDescription()
    
    def updateApiKeyVisibility(self):
        """Show only the API key field for the selected service."""
        apiServiceIndex = self.apiServiceChoice.GetSelection()
        
        # Hide all API key fields first
        self.openaiApiKeySizer.ShowItems(False)
        self.anthropicApiKeySizer.ShowItems(False)
        self.geminiApiKeySizer.ShowItems(False)
        self.cohereApiKeySizer.ShowItems(False)
        
        # Show only the relevant API key field
        if apiServiceIndex == 0:  # OpenAI
            self.openaiApiKeySizer.ShowItems(True)
        elif apiServiceIndex == 1:  # Anthropic
            self.anthropicApiKeySizer.ShowItems(True)
        elif apiServiceIndex == 2:  # Google
            self.geminiApiKeySizer.ShowItems(True)
        elif apiServiceIndex == 3:  # Cohere
            self.cohereApiKeySizer.ShowItems(True)
        
        # Refresh the layout
        self.Layout()
    
    def updateModelChoices(self):
        """Update the model choices based on the selected API service."""
        self.modelChoices.clear()
        
        apiServiceIndex = self.apiServiceChoice.GetSelection()
        if apiServiceIndex == 0:  # OpenAI
            self.modelChoices.extend(MODEL_OPTIONS["openai"])
        elif apiServiceIndex == 1:  # Anthropic
            self.modelChoices.extend(MODEL_OPTIONS["anthropic"])
        elif apiServiceIndex == 2:  # Google
            self.modelChoices.extend(MODEL_OPTIONS["gemini"])
        elif apiServiceIndex == 3:  # Cohere
            self.modelChoices.extend(MODEL_OPTIONS["cohere"])
        
        # Update the choice control if it exists
        if hasattr(self, 'modelChoice'):
            self.modelChoice.Clear()
            self.modelChoice.AppendItems(self.modelChoices)
            self.updateModelSelection()
    
    def updateModelSelection(self):
        """Update the selected model in the choice control."""
        selectedModel = config.conf["aiAssistant"]["selectedModel"]
        
        # If no model is selected or the selected model is not in the list
        if not selectedModel or selectedModel not in self.modelChoices:
            # Select the first model if available
            if self.modelChoices:
                self.modelChoice.SetSelection(0)
                return
        
        # Otherwise, select the model that matches the configured one
        for i, model in enumerate(self.modelChoices):
            if model == selectedModel:
                self.modelChoice.SetSelection(i)
                break
    
    def updateModelDescription(self):
        """Update the model description text."""
        modelIndex = self.modelChoice.GetSelection()
        if modelIndex >= 0 and modelIndex < len(self.modelChoices):
            model = self.modelChoices[modelIndex]
            # Add model descriptions here
            descriptions = {
                # OpenAI
                "gpt-4-turbo": "OpenAI's advanced model with knowledge cutoff in April 2023.",
                "gpt-4-omni": "OpenAI's multimodal model that's more efficient than previous GPT-4 versions.",
                "gpt-4": "OpenAI's original GPT-4 model.",
                
                # Anthropic
                "claude-3-5-sonnet-20240620": "Anthropic's improved model with enhanced reasoning capabilities.",
                "claude-3-opus-20240229": "Anthropic's most powerful model for highly complex tasks.",
                "claude-3-sonnet-20240229": "Anthropic's model with ideal balance of intelligence and speed.",
                "claude-3-haiku-20240307": "Anthropic's fastest and most compact model for near-instant responsiveness.",
                
                # Google
                "gemini-1.5-pro": "Google's mid-size multimodal model optimized for complex reasoning tasks.",
                "gemini-1.5-flash": "Google's smaller, faster model for simpler tasks.",
                "gemini-2.0-flash": "Google's newest flash model with improved capabilities.",
                "gemini-2.0-flash-lite-preview": "A lighter version of Gemini 2.0 Flash optimized for efficiency.",
                
                # Cohere
                "command-a-03-2025": "Cohere's Command A model for generation and reasoning tasks.",
            }
            
            if model in descriptions:
                self.modelDescription.SetLabel(descriptions[model])
            else:
                self.modelDescription.SetLabel("")
        else:
            self.modelDescription.SetLabel("")
        
        self.Layout()
    
    def onApiServiceChange(self, evt):
        """Handle API service change"""
        self.updateApiKeyVisibility()
        self.updateModelChoices()
        self.updateModelDescription()
    
    def onSave(self):
        """Save the settings."""
        # Save API keys
        config.conf["aiAssistant"]["openaiApiKey"] = self.openaiApiKeyEdit.GetValue()
        config.conf["aiAssistant"]["anthropicApiKey"] = self.anthropicApiKeyEdit.GetValue()
        config.conf["aiAssistant"]["geminiApiKey"] = self.geminiApiKeyEdit.GetValue()
        config.conf["aiAssistant"]["cohereApiKey"] = self.cohereApiKeyEdit.GetValue()
        
        # Save API service selection
        serviceIndex = self.apiServiceChoice.GetSelection()
        if serviceIndex == 0:
            config.conf["aiAssistant"]["apiService"] = "openai"
        elif serviceIndex == 1:
            config.conf["aiAssistant"]["apiService"] = "anthropic"
        elif serviceIndex == 2:
            config.conf["aiAssistant"]["apiService"] = "gemini"
        elif serviceIndex == 3:
            config.conf["aiAssistant"]["apiService"] = "cohere"
        
        # Save selected model
        modelIndex = self.modelChoice.GetSelection()
        if 0 <= modelIndex < len(self.modelChoices):
            config.conf["aiAssistant"]["selectedModel"] = self.modelChoices[modelIndex]
        
        # Save max tokens
        config.conf["aiAssistant"]["maxTokens"] = self.maxTokensEdit.GetValue()
        
        # Save temperature
        config.conf["aiAssistant"]["temperature"] = self.temperatureSlider.GetValue() / 100.0
        
        # Save speech option
        config.conf["aiAssistant"]["read_response"] = self.speechCheckbox.IsChecked()


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    scriptCategory = "AI Assistant"
    
    def __init__(self):
        super(GlobalPlugin, self).__init__()
        
        # Set up config
        config.conf.spec["aiAssistant"] = SPEC
        
        # Set default model if not already set
        if not config.conf['aiAssistant']['selectedModel']:
            service = config.conf['aiAssistant']['apiService']
            if service in MODEL_OPTIONS and MODEL_OPTIONS[service]:
                config.conf['aiAssistant']['selectedModel'] = MODEL_OPTIONS[service][0]
        
        # Add settings panel to NVDA settings
        gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(AiAssistantSettingsPanel)
        
        # Flag to prevent multiple simultaneous requests
        self.processing = False
    
    def terminate(self):
        # Remove settings panel when terminating the plugin
        try:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(AiAssistantSettingsPanel)
        except ValueError:
            pass
        super(GlobalPlugin, self).terminate()
    
    @scriptHandler.script(
        description="Sends selected text to the configured AI model and presents the response",
        gesture="kb:control+shift+k"
    )
    def script_processWithAI(self, gesture):
        # Prevent multiple simultaneous requests
        if self.processing:
            ui.message("Processing previous request. Please wait.")
            return
        
        # Get selected text
        selectedText = self.getSelectedText()
        if not selectedText:
            ui.message("No text selected. Please select text before using this command.")
            return
        
        # Check if API key is available
        apiService = config.conf["aiAssistant"]["apiService"]
        serviceKeyMap = {
            "openai": "openaiApiKey",
            "anthropic": "anthropicApiKey",
            "gemini": "geminiApiKey",
            "cohere": "cohereApiKey"
        }
        
        api_key = config.conf["aiAssistant"].get(serviceKeyMap.get(apiService, ""))
        
        if not api_key:
            ui.message(f"API key not configured for {apiService}. Please set up your API key in NVDA settings.")
            return
        
        # Get model
        selectedModel = config.conf["aiAssistant"]["selectedModel"]
        if not selectedModel or selectedModel not in MODEL_OPTIONS.get(apiService, []):
            selectedModel = MODEL_OPTIONS.get(apiService, [""])[0]
        
        # Announce that processing is starting
        ui.message(f"Processing request with {selectedModel}...")
        
        self.processing = True
        # Start processing in a background thread
        threading.Thread(
            target=self.processRequest,
            args=(selectedText, apiService, selectedModel, api_key),
            daemon=True
        ).start()
    
    def getSelectedText(self) -> str:
        """Get the currently selected text."""
        try:
            # First try to get selection from focus object
            focus = api.getFocusObject()
            if hasattr(focus, "selection") and focus.selection:
                return focus.selection.text
            
            # If not available from focus, try getting it from clipboard
            # First store the original clipboard content
            original = api.getClipData()
            
            # Send Ctrl+C to copy selection
            wx.CallAfter(lambda: self.sendCopyCommand())
            
            # Wait a bit for the clipboard to be updated
            wx.CallLater(100, lambda: self.getClipboardText(original))
            return ""  # Placeholder, will be updated by getClipboardText
            
        except Exception as e:
            log.error(f"Error getting selected text: {e}")
            return ""
    
    def sendCopyCommand(self):
        """Send Ctrl+C to copy selection to clipboard."""
        import keyboardHandler
        keyboardHandler.KeyboardInputGesture.fromName("control+c").send()
    
    def getClipboardText(self, original):
        """Get text from clipboard and restore original content."""
        try:
            # Get the current clipboard content
            clipData = api.getClipData()
            
            # If it's different from the original, it's our selection
            if clipData != original and clipData:
                # Process the selected text
                self.processRequest(
                    clipData,
                    config.conf["aiAssistant"]["apiService"],
                    config.conf["aiAssistant"]["selectedModel"],
                    self.getApiKey()
                )
            else:
                ui.message("No text selected. Please select text before using this command.")
                self.processing = False
            
            # Restore the original clipboard content if needed
            if original:
                api.copyToClip(original)
        except Exception as e:
            log.error(f"Error getting clipboard text: {e}")
            ui.message("Error getting selected text. Please try again.")
            self.processing = False
    
    def getApiKey(self) -> str:
        """Get the API key for the selected provider."""
        apiService = config.conf["aiAssistant"]["apiService"]
        serviceKeyMap = {
            "openai": "openaiApiKey",
            "anthropic": "anthropicApiKey",
            "gemini": "geminiApiKey",
            "cohere": "cohereApiKey"
        }
        
        return config.conf["aiAssistant"].get(serviceKeyMap.get(apiService, ""), "")
    
    def processRequest(self, text: str, apiService: str, model: str, api_key: str):
        """Process the request with the selected AI model."""
        try:
            # Call the appropriate API based on the provider
            if apiService == 'openai':
                result = self.callOpenAI(text, model, api_key)
            elif apiService == 'anthropic':
                result = self.callAnthropic(text, model, api_key)
            elif apiService == 'gemini':
                result = self.callGemini(text, model, api_key)
            elif apiService == 'cohere':
                result = self.callCohere(text, model, api_key)
            else:
                result = "Unsupported provider."
            
            self.announceResult(result)
            
        except Exception as e:
            log.error(f"Error processing request: {e}")
            self.announceResult(f"Error processing request: {str(e)}")
        finally:
            self.processing = False
    
    def callOpenAI(self, text: str, model_id: str, api_key: str) -> str:
        """Call the OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": config.conf["aiAssistant"]["maxTokens"],
            "temperature": config.conf["aiAssistant"]["temperature"]
        }
        
        return self.makeRequest(url, headers, data)
    
    def callAnthropic(self, text: str, model_id: str, api_key: str) -> str:
        """Call the Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model_id,
            "max_tokens": config.conf["aiAssistant"]["maxTokens"],
            "messages": [{"role": "user", "content": text}],
            "temperature": config.conf["aiAssistant"]["temperature"]
        }
        
        return self.makeRequest(url, headers, data)
    
    def callGemini(self, text: str, model_id: str, api_key: str) -> str:
        """Call the Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "temperature": config.conf["aiAssistant"]["temperature"],
                "maxOutputTokens": config.conf["aiAssistant"]["maxTokens"],
            }
        }
        
        return self.makeRequest(url, headers, data)
    
    def callCohere(self, text: str, model_id: str, api_key: str) -> str:
        """Call the Cohere API."""
        url = "https://api.cohere.ai/v1/chat"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_id,
            "message": text,
            "max_tokens": config.conf["aiAssistant"]["maxTokens"],
            "temperature": config.conf["aiAssistant"]["temperature"]
        }
        
        return self.makeRequest(url, headers, data)
    
    def makeRequest(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> str:
        """Make a request to an API."""
        try:
            # Convert data to JSON
            data_bytes = json.dumps(data).encode('utf-8')
            
            # Create request
            request = urllib.request.Request(url, data=data_bytes, headers=headers)
            
            # Send request
            with urllib.request.urlopen(request, timeout=30) as response:
                response_data = response.read()
                
            # Parse response
            response_json = json.loads(response_data)
            
            # Extract content based on API
            if "choices" in response_json:  # OpenAI
                if "message" in response_json["choices"][0]:
                    # Chat completions format
                    return response_json["choices"][0]["message"]["content"]
                else:
                    # Older completions format
                    return response_json["choices"][0]["text"]
                    
            elif "content" in response_json:  # Anthropic
                if isinstance(response_json["content"], list):
                    return response_json["content"][0]["text"]
                return response_json["content"]
                
            elif "candidates" in response_json:  # Gemini
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
                
            elif "text" in response_json:  # Cohere (older format)
                return response_json["text"]
            
            elif "message" in response_json:  # Cohere (newer chat format)
                if isinstance(response_json["message"], dict) and "text" in response_json["message"]:
                    return response_json["message"]["text"]
                elif isinstance(response_json["message"], dict) and "content" in response_json["message"]:
                    return response_json["message"]["content"]
                
            # Unknown format, return raw response
            return str(response_json)
            
        except urllib.error.HTTPError as e:
            error_message = e.read().decode('utf-8')
            try:
                error_json = json.loads(error_message)
                if "error" in error_json:
                    error_detail = error_json.get("error", {}).get("message", str(e))
                    raise Exception(f"API error: {error_detail}")
            except json.JSONDecodeError:
                pass
            raise Exception(f"HTTP error: {e.code} - {error_message}")
            
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def announceResult(self, result: str):
        """Announce the result of the request."""
        if not result:
            result = "No response received from the AI model."
        
        # Clean up the result to remove excessive newlines and spaces
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {3,}', '  ', result)
        
        # Clean markdown formatting
        result = clean_markdown(result)
        
        # Use TextWindow to display the result instead of browseable message
        wx.CallAfter(lambda: TextWindow(
            result, 
            "Message", 
            readOnly=True
        ))
        
        # If read_response is enabled, also speak the result
        if config.conf["aiAssistant"]["read_response"]:
            wx.CallAfter(speech.speakMessage, result)