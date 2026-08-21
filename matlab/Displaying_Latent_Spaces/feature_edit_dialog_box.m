
function out = feature_edit_dialog_box(prompt, title, defaultText, defaultChecked)
% out = inputWithCheckbox(prompt, title, defaultText, defaultChecked)
% prompt         - prompt string shown above the edit field (default: 'Enter value:')
% title          - dialog window title (default: 'Input')
% defaultText    - default edit text (default: '')
% defaultChecked - logical default for checkbox (default: false)
% out is struct: out.text (char) and out.checked (logical). If cancelled, out = [].

if nargin < 1 || isempty(prompt), prompt = 'Enter value:'; end
if nargin < 2 || isempty(title),  title = 'Input'; end
if nargin < 3, defaultText = ''; end
if nargin < 4, defaultChecked = false; end

% Create UI figure and components
fig = uifigure('Name', title, 'Position', [300 300 360 160], 'Resize', 'off', 'WindowStyle', 'modal');

pad = 12;
 lbl = uilabel(fig, 'Text', prompt, 'Position', [pad 110 336 24]);

 edt = uieditfield(fig, 'text', 'Value', defaultText, 'Position', [pad 80 336 24]);

 cbx = uicheckbox(fig, 'Text', 'Plot manual calls associated with other DASARs?', 'Value', logical(defaultChecked), ...
                  'Position', [pad 50 400 22]);

 btnOK = uibutton(fig, 'Text', 'OK', 'Position', [200 12 70 28], 'ButtonPushedFcn', @okCB);
 btnCancel = uibutton(fig, 'Text', 'Cancel', 'Position', [280 12 70 28], 'ButtonPushedFcn', @cancelCB);

% Store result in figure UserData, use uiwait/ uiresume for modal behavior
fig.UserData = struct('Choice','cancel','Text',defaultText,'Checked',logical(defaultChecked));
movegui(fig,'center');
uiwait(fig);

ud = fig.UserData;
if isequal(ud.Choice,'ok')
    out.text = ud.Text;
    out.checked = ud.Checked;
    out.index=str2num(out.text);
    if out.checked
        out.index(2)=out.index(1);  %Ensure that only one index is plottable.
    end

else
    out = [];
end
delete(fig);

% Callbacks
    function okCB(~,~)
        fig.UserData.Choice = 'ok';
        fig.UserData.Text = edt.Value;
        fig.UserData.Checked = cbx.Value;
        uiresume(fig);
    end

    function cancelCB(~,~)
        fig.UserData.Choice = 'cancel';
        uiresume(fig);
    end
end