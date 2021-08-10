import os
workspace ="""
{
	"folders": [
		{
			"path": "chapter02"
		},
		{
			"path": "download/CodeSamples/chapter$num"
		},
		{
			"path": "download/CUDA_C_J_Samples/$num/chapter$num"
		}
	],
	"settings": {
		"C_Cpp.clang_format_style": "{ BasedOnStyle: LLVM, BreakBeforeBraces: Attach, SpaceBeforeParens: Never, IndentWidth: 4 }", //c,cpp,cu format style
		"code-runner.executorMapByFileExtension": {
			".cu": "cd $dir && nvcc $fileName --generate-code arch=compute_61,code=sm_61 --generate-code arch=compute_75,code=sm_75 -o $fileNameWithoutExt.out && ./$fileNameWithoutExt.out", //code runner 
		},
		"code-runner.runInTerminal": true,
	},
	"extensions": {
		"recommendations": [
			"ms-vscode.cpptools", //c,cpp,cu formatter and linter
			"davidanson.vscode-markdownlint", //markdown linter
			"yzhang.markdown-all-in-one", // markdown supporter
			"docsmsft.docs-markdown", // markdown supporter by microsoft
			"ms-ceintl.vscode-language-pack-ja", // japanese language pack
			"formulahendry.code-runner", // code executer
			"ms-python.python", // python
			"tabnine.tabnine-vscode", // good advisor for coding
		]
	}
}
"""
def chapterize(a:int):
    return str(a).zfill(2)
for chapter_number in map(chapterize,range(1,11)):
    os.makedirs(f"chapter{chapter_number}",exist_ok=True)
    with open(f"chapter{chapter_number}.code-workspace",mode="w") as f:
        f.write(workspace.replace("$num",chapter_number))