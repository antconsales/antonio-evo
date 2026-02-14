import React from 'react';
import { Search, FileText, Code, Image, FolderOpen, Wrench } from 'lucide-react';
import { useTranslation } from '../i18n';

const TOOL_ICONS = {
  web_search: Search,
  read_file: FileText,
  write_file: FileText,
  list_directory: FolderOpen,
  execute_code: Code,
  analyze_image: Image,
};

const TOOL_I18N_KEYS = {
  web_search: 'tools.searchingWeb',
  read_file: 'tools.readingFile',
  write_file: 'tools.writingFile',
  list_directory: 'tools.listingDirectory',
  execute_code: 'tools.executingCode',
  analyze_image: 'tools.analyzingImage',
};

function ToolActionIndicator({ actions = [] }) {
  const { t } = useTranslation();

  if (actions.length === 0) return null;

  return (
    <div className="tool-actions">
      {actions.map((action, index) => {
        const IconComponent = TOOL_ICONS[action.tool] || Wrench;
        const i18nKey = TOOL_I18N_KEYS[action.tool] || 'tools.usingTool';
        const label = t(i18nKey) || action.tool;

        // Build detail string from arguments
        let detail = '';
        if (action.arguments) {
          if (action.arguments.query) detail = action.arguments.query;
          else if (action.arguments.path) detail = action.arguments.path;
          else if (action.arguments.image_path) detail = action.arguments.image_path;
          else if (action.arguments.code) detail = action.arguments.code.substring(0, 60);
        }

        return (
          <div
            key={`${action.tool}-${index}`}
            className={`tool-action ${action.status === 'done' ? 'tool-action-done' : 'tool-action-active'}`}
          >
            <div className="tool-action-icon">
              <IconComponent size={14} />
            </div>
            <span className="tool-action-label">{label}</span>
            {detail && <span className="tool-action-detail">{detail}</span>}
            {action.status !== 'done' && (
              <div className="tool-action-spinner" />
            )}
            {action.status === 'done' && (
              <span className={`tool-action-result ${action.success ? '' : 'tool-action-failed'}`}>
                {action.success ? '\u2713' : '\u2717'}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default ToolActionIndicator;
