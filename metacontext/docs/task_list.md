# 📋 Task List: GitHub Copilot Integration

## 🎯 **Objective**
Integrate GitHub Copilot as a code companion provider to leverage existing prompt templates while utilizing IDE context instead of manual codebase scanning.

## 🏗️ **Phase 1: Architecture Foundation**

### 1.1 Create Companion Provider Interface
- [x] **Create `src/metacontext/ai/handlers/companions/companion_provider.py`**
  - [x] Define `BaseCompanionProvider` abstract class
  - [x] Implement common interface matching `BaseLLMProvider`
  - [x] Add methods: `is_available()` (combines detection + availability check), `generate_prompt()`, `wait_for_response()`
  - [x] Create `TemplateContext` dataclass for type-safe prompt generation parameters
  - [x] Include clipboard automation utilities (`pyperclip` integration)

### 1.2 Implement GitHub Copilot Detection
- [x] **Create `src/metacontext/ai/handlers/companions/copilot_provider.py`**
  - [x] Detect GitHub Copilot availability via `gh copilot --version`
  - [x] Detect VS Code with Copilot extension installed
  - [x] Check for authenticated GitHub CLI session
  - [x] Implement fallback detection methods

### 1.3 Update Provider Factory
- [x] **Create `src/metacontext/ai/handlers/companions/companion_factory.py`**
  - [x] Add `CompanionProviderFactory` class
  - [x] Implement companion detection logic (priority order)
  - [x] Add graceful fallback to API providers
  - [x] Include configuration for forced provider selection

## 🧩 **Phase 2: Template Adaptation System** 

### 2.1 Create Companion Template Adapter (Modular Design)
- [x] **Create `src/metacontext/ai/handlers/companions/template_adapter.py`**
  - [x] `CompanionTemplateAdapter` class with broken-down methods for testability
  - [x] Method: `load_api_template()` - loads and parses YAML templates (20-30 lines)
  - [x] Method: `sanitize_api_elements()` - strips JSON schema requirements and API-specific instructions (20-30 lines)
  - [x] Method: `convert_schema_to_yaml()` - converts Pydantic schemas to YAML output format (20-30 lines)
  - [x] Method: `generate_companion_prompt()` - assembles final companion prompt (20-30 lines)
  - [x] Helper functions: `create_tabular_companion_prompt()`, `create_model_companion_prompt()`, etc.
  - [x] Defer `_extract_schema_structure()` to later milestone (not needed for MVP)

### 2.2 Template Adaptation Logic
- [x] **Reuse existing YAML templates from `src/metacontext/ai/prompts/templates/`**
  - [x] Load: `templates/tabular/column_analysis.yaml`
  - [x] Load: `templates/model/model_analysis.yaml` 
  - [x] Load: `templates/media/media_analysis.yaml`
  - [x] Load: `templates/geospatial/vector_analysis.yaml` and `raster_analysis.yaml`

- [x] **Template modification functions**
  - [x] Remove JSON schema requirements (`${schema_json}`, validation rules)
  - [x] Reduce unnecessary character limits and efficiency constraints
  - [x] Remove API-specific output formatting requirements
  - [x] Preserve core forensic analysis instructions and investigative approach
  - [x] Add workspace-wide analysis instructions ("examine ALL files in workspace")

### 2.3 Output Structure Preservation
- [x] **Maintain metacontext compatibility**
  - [x] Convert Pydantic schema classes to YAML output templates
  - [x] Ensure companions return data in exact metacontext structure
  - [x] Add clear instructions for saving response as `metacontext_response.yaml`
  - [x] Include field validation and structure requirements
  - [x] Preserve all schema field names and types

### 2.4 Context Enhancement for Companions
- [x] **Leverage IDE workspace awareness**
  - [x] Replace manual file scanning with "analyze entire workspace" instructions
  - [x] Add prompts for cross-file relationship analysis
  - [x] Include code intelligence utilization (imports, dependencies, references)
  - [x] Add semantic knowledge extraction from workspace context
  - [x] Remove token limit constraints (companions can handle larger contexts)

## ⚙️ **Phase 3: Copilot Provider Implementation**

### 3.1 Core Copilot Provider
- [x] **Implement `GitHubCopilotProvider` class**
  - [x] Method: `is_available()` - check Copilot installation
  - [x] Method: `generate_prompt()` - format prompts for Copilot chat using `TemplateContext`
  - [x] Method: `send_to_copilot()` - clipboard automation with fallback for systems without VS Code
  - [x] Method: `wait_for_response()` - simple manual file drop parsing (`response.yaml`) - defer file watchers
  - [x] Method: `parse_response()` - extract structured data from YAML response
  - [x] Make clipboard interaction optional (users can paste manually)
  - [x] **ENHANCED: Platform-specific VS Code detection (macOS, Windows, Linux)**
  - [x] **ENHANCED: Comprehensive environment validation with detailed diagnostics**
  - [x] **ENHANCED: User-friendly error handling system**
    - [x] `get_detailed_diagnostics()` - comprehensive diagnostic information
    - [x] `format_user_friendly_error()` - clear, actionable error messages
    - [x] Platform-specific installation instructions (macOS, Windows, Linux)
    - [x] Multiple solution approaches (GUI + CLI methods)
    - [x] Intelligent issue detection and categorization
    - [x] Step-by-step troubleshooting guides

### 3.2 IDE Integration Utilities (Simplified)
- [x] **Create `src/metacontext/ai/handlers/companions/ide_automation.py`**
  - [x] Function: `detect_vscode()` - check if VS Code is running
  - [x] Function: `copy_to_clipboard()` - cross-platform clipboard (optional automation)
  - [x] Function: `focus_vscode()` - bring VS Code to front (macOS/Windows/Linux)
  - [x] Function: `create_response_file()` - temp file for user responses
  - [x] Function: `get_ide_environment_info()` - environment capability assessment
  - [x] Class: `IDEAutomation` - clean OOP interface for IDE automation
  - [x] **ENHANCED: Cross-platform process detection and window focus**
  - [x] **ENHANCED: Comprehensive environment validation and capabilities**
  - [x] Defer watchdog file monitoring to post-MVP ✅

### 3.3 Response Processing & YAML Parsing
- [x] **Create response parsing utilities in `CompanionTemplateAdapter`**
  - [x] Function: `parse_companion_response()` - load YAML response files  
  - [x] Function: `validate_response_structure()` - ensure metacontext schema compliance
  - [x] Function: `convert_yaml_to_pydantic()` - convert YAML back to Pydantic model instances
  - [x] Integration with existing schema validation pipeline
  - [x] Error handling for malformed or incomplete responses

## 🔄 **Phase 4: Handler Integration**

### 4.1 Update Handler Generate Context Methods (Preserve Context Quality)
- [x] **Modify `src/metacontext/handlers/tabular.py`**
  - [x] Add `companion_mode` parameter to `generate_context()`
  - [x] Import and use `CompanionTemplateAdapter` from `ai.handlers.companions.template_adapter`
  - [x] Generate companion prompts using adapted templates: `load_api_template(templates/tabular/column_analysis.yaml)`
  - [x] Keep lightweight deterministic analysis (schema + file structure hints) for Copilot grounding
  - [x] Skip content flattening but preserve essential context for quality
  - [x] Add response parsing to convert YAML back to Pydantic schemas

- [x] **Modify `src/metacontext/handlers/model.py`**
  - [x] Add companion prompt generation using `templates/model/model_analysis.yaml`
  - [x] Include minimal deterministic summary (model type, file size) for context
  - [x] Let Copilot discover training scripts via workspace analysis
  - [x] Format model analysis requests for chat interface while preserving output structure

- [x] **Modify `src/metacontext/handlers/media.py`**
  - [x] Add companion-specific media analysis using `templates/media/media_analysis.yaml` 
  - [x] Include basic file metadata (type, size) for Copilot context
  - [x] Let Copilot analyze content based on filename/workspace context
  - [x] Ensure media analysis maintains metacontext schema compatibility

- [x] **Modify `src/metacontext/handlers/geospatial.py`**
  - [x] Add companion prompts using `templates/geospatial/vector_analysis.yaml` and `raster_analysis.yaml`
  - [x] Include basic geospatial metadata (CRS, bounds) for context
  - [x] Maintain raster vs vector detection logic for appropriate template selection
  - [x] Let Copilot handle detailed spatial relationship analysis

### 4.2 Update Metacontextualize Core
- [x] **Modify `src/metacontext/metacontextualize.py`**
  - [x] Add companion provider detection
  - [x] Add `--companion` CLI flag (already implemented in CLI)
  - [x] Route to companion workflow when available
  - [x] Maintain compatibility with API workflow
  - [ ] Maintain compatibility with API workflow

## 🧪 **Phase 5: Testing & Validation**

### 5.1 Unit Tests (Focused)
- [ ] **Create `tests/test_companion_provider.py`**
  - [ ] Test companion detection logic
  - [ ] Test template adapter modular methods (`load_api_template`, `sanitize_api_elements`, etc.)
  - [ ] Test response parsing utilities
  - [ ] Stub IDE automation (don't mock on day one)

### 5.2 Integration Tests (Simplified)
- [ ] **Create `tests/test_copilot_integration.py`**
  - [ ] Focus on adapter logic and detection flow
  - [ ] Test template conversion validation
  - [ ] Defer cross-platform automation testing to post-MVP
  - [ ] Template quality validation tests

### 5.3 Manual Testing Scenarios
- [ ] **Create testing checklist**
  - [ ] Test with VS Code + Copilot on macOS/Windows/Linux
  - [ ] Test fallback to API when Copilot unavailable
  - [ ] Test template quality and response usefulness
  - [ ] Validate semantic knowledge integration

## 🚀 **Phase 6: CLI & UX Enhancements**

### 6.1 CLI Updates (MVP Focus)
- [ ] **Add companion-specific commands**
  - [ ] `metacontext --companion detect` - show available companions
  - [ ] `metacontext --companion copilot <file>` - analyze with Copilot
  - [ ] `metacontext --force-api` - skip companion detection
  - [ ] Defer `--companion-bridge` until actual IPC with VS Code is needed (post-MVP)

### 6.2 User Experience
- [ ] **Create `docs/companion_usage.md`**
  - [ ] Setup instructions for GitHub Copilot
  - [ ] Workflow examples and best practices
  - [ ] Troubleshooting common issues
  - [ ] Performance comparison with API mode

### 6.3 Interactive Workflow
- [ ] **Create companion workflow interface**
  - [ ] Progress indicators during prompt generation
  - [ ] Clear instructions for user actions
  - [ ] Response validation and retry options
  - [ ] Automatic result integration

## 📚 **Phase 7: Documentation & Examples**

### 7.1 Architecture Documentation
- [ ] **Update `docs/architecture_guides.md`**
  - [ ] Explain companion vs API provider architecture
  - [ ] Document ProviderFactory pattern
  - [ ] Include decision tree for provider selection

### 7.2 Usage Examples
- [ ] **Create `examples/companion_usage/`**
  - [ ] Basic Copilot integration example
  - [ ] Advanced prompt customization
  - [ ] Hybrid API + companion workflows
  - [ ] Performance optimization tips

### 7.3 Migration Guide (Post-MVP)
- [ ] **Defer migration documentation until working prototype**
  - [ ] How to adapt existing workflows
  - [ ] Benefits of companion vs API mode
  - [ ] When to use each approach

## 🔧 **Technical Implementation Details**

### Dependencies to Add
```toml
# pyproject.toml additions
pyperclip = "^1.8.2"  # Clipboard automation (optional)
psutil = "^5.9.0"     # Process detection
# Defer: watchdog = "^3.0.0"   # File watching (post-MVP)
```

### Key Files to Create/Modify
```
src/metacontext/ai/handlers/companions/
├── template_adapter.py             # NEW: Template adaptation system
├── copilot_provider.py             # ✅ COMPLETE: GitHub Copilot implementation  
├── companion_provider.py           # ✅ COMPLETE: Base companion interface
├── companion_factory.py            # ✅ COMPLETE: Provider selection logic
└── ide_automation.py               # NEW: IDE interaction utilities

# NO new template directories needed! 
# Reuse existing: src/metacontext/ai/prompts/templates/

tests/companion/                    # NEW: Companion-specific tests
├── test_copilot_provider.py
├── test_template_adapter.py        # NEW: Test template adaptation
└── test_ide_automation.py
```

## 🎯 **Success Criteria**

### MVP (Minimum Viable Product)
- [ ] GitHub Copilot detection works across platforms
- [ ] Basic prompt generation and clipboard automation
- [ ] At least tabular analysis working with Copilot
- [ ] Graceful fallback to API providers

### Full Feature Complete
- [ ] All four handlers support companion mode with template adaptation
- [ ] Template adaptation maintains analysis quality and output structure compatibility  
- [ ] Cross-platform IDE automation
- [ ] YAML response parsing with Pydantic schema validation
- [ ] Comprehensive documentation and examples

### Performance Goals  
- [ ] Companion workflow < 30 seconds per analysis
- [ ] Template adaptation preserves semantic knowledge quality and output structure
- [ ] User experience feels "almost automated"
- [ ] 100% schema compatibility between API and companion modes
- [ ] YAML ↔ Pydantic conversion maintains data fidelity

## 🏁 **Implementation Priority**

**Week 1-2**: Phase 1 (Architecture Foundation)  
**Week 3-4**: Phase 2 (Template Adaptation System)  
**Week 5-6**: Phase 3 (Copilot Provider)  
**Week 7-8**: Phase 4 (Handler Integration)  
**Week 9**: Phase 5 (Testing)  
**Week 10**: Phase 6-7 (UX & Documentation)

---

**🔥 This integration will make metacontext the first tool to seamlessly bridge IDE code companions with structured metadata generation while maintaining 100% schema compatibility!**
