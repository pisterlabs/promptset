// The module 'vscode' contains the VS Code extensibility API
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const cp = require('child_process');

// This method is called when your extension is activated
// The extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
	// Code here will be executed only once when your extension is activated
	// Create a map to store output channels
	const outputChannels = new Map();

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with  registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('promptr.suggest_prompts', function () {
		// The code you place here will be executed every time your command is executed
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return; // No workspace is open
        }

		// Check if the output channel already exists
		let outputChannel = outputChannels.get("Promptr");
		if (outputChannel) {
			outputChannel.clear();  // Clear the output channel
		} else {
			// Create an output channel
			outputChannel = vscode.window.createOutputChannel("Promptr");
			outputChannels.set("Promptr", outputChannel);
			context.subscriptions.push(outputChannel);
		}
		outputChannel.appendLine('Consider the following prompts for your code:');
		// ... append more lines as needed
        
		// Recurse through the workspace to find Python files
		let python_files = [];
        workspaceFolders.forEach((folder) => {
			const rootPath = folder.uri.fsPath;
            findPythonFiles(rootPath, outputChannel, python_files);
        });

		// Creating Child process to grab prompts from all python files
		let pythonfiles_str = python_files.join("~~~");
		// console.log(pythonfiles_str);
		const pythonScriptPath = path.join(__dirname, 'get_prompts.py');
		// const pythonProcess = cp.spawn('python3', [pythonScriptPath, pythonfiles_str]);
		// pythonProcess.stdout.on('data', (data) => {
		// 	console.log(data.toString());
		// 	outputChannel.appendLine(data.toString());
		// 	outputChannel.show(true);  // Focus on the output channel
		// });
		cp.exec(`python3 ${pythonScriptPath} ${pythonfiles_str}`, (err, stdout, stderr) => {
			if (err) {
				console.error(err);
				return;
			}
			outputChannel.appendLine(stdout);
			outputChannel.show(true);  // Focus on the output channel
		});
		
		// Display a message box to the user
		vscode.window.showInformationMessage('Welcome to Promptr!');
		outputChannel.show();  // Focus on the output channel
	});

	// Add the disposable to the context so it can be cleaned up when the extension is deactivated
	context.subscriptions.push(disposable);
}

function findPythonFiles(dir, outputChannel, python_files) {
    fs.readdirSync(dir).forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
            findPythonFiles(filePath, outputChannel, python_files); // Recurse into a subdirectory
        } else if (filePath.endsWith('.py')) {
            python_files.push(filePath);         
        }
    });
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}

