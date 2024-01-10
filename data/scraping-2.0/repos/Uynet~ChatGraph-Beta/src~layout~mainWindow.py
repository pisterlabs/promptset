import os

import openai
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAction, QFileDialog, QMainWindow, QMenu,
                             QMessageBox, QSplitter)
from editors.controllers.actions.cgpAction import CgpAction

from editors.controllers.mainWindowController import MainWindowController
from editors.types.dataType import GraphData
from editors.views.nodeScene import NodeScene
from layout.editorpanel import EditorPanel, LeftPanel, RightPanel
from utils.fileLoader import FileLoader
from utils.sound import CSound
from utils.styles import Styles
from utils.util import Util
from editors.models.serializer import Serializer


class MainWindow(QMainWindow):
    # クラス変数 _instance を None で初期化
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super(MainWindow,self).__init__()
        # todo : toGraphViews
        self.controller = MainWindowController(self)
        self.lastOpenFile = Util.getDefaultFilePath()
        self.initUI()

    
    def addTab(self, widget, title):
        self.editorPanel.addTab(widget, title)
        
    def closeTab(self, index):
        self.editorPanel.removeTab(index)

    def getNodeScene(self):
        return self.nodeScene

    def initMenu(self):
        self.menuBarWidget = self.menuBar()
        self.menuBarWidget.setStyleSheet(Styles.qClass("QMenuBar","QMenuBar"))
        self.fileMenuWidget = self.menuBarWidget.addMenu("File")
        self.fileMenuWidget.setStyleSheet(Styles.qClass("windowMenu","QMenu"))

        actions = [
            {"text": "シーンを削除", "action": self.clean},
            {"text": "開く", "action": self.openGraphFile},
            {"text": "保存(.json)", "action": self.saveGraphFile , "shortcut":"Ctrl+S"},
            {"text": "読み込み", "action": self.importGraphFile},
            # {"text": "再読み込み", "action": self.reload , "shortcut":"Ctrl+R"},
            # {"text": "モジュール読み込み(.cgp)", "action": self.importModule},
            # {"text": "モジュールとして保存(.cgp)", "action": self.saveModuleFile},
        ]

        for action in actions:
            actionObj = QAction(action["text"], self)
            actionObj.triggered.connect(action["action"])
            self.fileMenuWidget.addAction(actionObj)
            # key bind to ctrl + s
            if action["text"] == "save graph (.json)":
                self.save_action = actionObj

            keyBinding = action.get("shortcut")
            if keyBinding is not None:
                actionObj.setShortcut(keyBinding)

    def createMenu(self, parent, name, actions):
        menu = QMenu(name,self)
        for action in actions:
            menu.addAction(action)
        return menu

    def initUI(self):
        self.nodeScene = NodeScene() 

         # メニューバーの作成
        self.initMenu()
        self.menuBarWidget = self.menuBar()
        self.menuBarWidget.setStyleSheet(Styles.qClass("QMenuBar","QMenuBar"))
        # settings
        self.settingsMenuWidget = self.menuBarWidget.addMenu("Settings")
        apia  = QAction("API Key", self)
        apia.triggered.connect(self.settingApiKey)
        self.settingsMenuWidget.addAction(apia)
        self.settingsMenuWidget.setStyleSheet(Styles.qClass("windowMenu","QMenu"))

        self.leftPanel= LeftPanel(self , self.nodeScene)
        # ノードの編集画面とか
        self.editorPanel = EditorPanel(self , self.nodeScene)
        # chatとinspectorなど
        self.rightPanel = RightPanel(self , self.nodeScene)

        # QSplitterを作成し、タブウィジェットを追加
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.leftPanel)
        self.splitter.addWidget(self.editorPanel)
        self.splitter.addWidget(self.rightPanel)

        # ウィンドウにスプリッターを追加
        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([200, 600, 200])
        s = Styles.qClass("QSplitter","QSplitter")
        self.splitter.setStyleSheet(s)
        self.setGeometry(0, 0, 1400, 800)
        # full sc
        self.showMaximized()

        # 背景色
        self.setStyleSheet(Styles.className("mainWindow"))

    def addThread(self,thread):
        self.thread = thread 
        self.thread.start()

    ####### FILES ##########

    def saveGraphFile(self):
        filepath , _= self.openFileDialog(False)
        if filepath == None: return
        nodeGraph = self.nodeScene.nodeGraph
        data = nodeGraph.SceneToGraphData()
        FileLoader.write(data , filepath)
        return

    def saveModuleFile(self ):
        filePath , _= self.openFileDialog(isLoad = False, isModule = True)
        if filePath == None: return
        nodeGraph = self.nodeScene.nodeGraph
        data = nodeGraph.SceneToGraphData()
        FileLoader.write(data , filePath)
        return

    def openGraphFile(self):
        # 読み込み時に初期化の警告を出し、キャンセルなら中断
        isCancelled = not self.clean()
        if isCancelled:return
        filepath , _= self.openFileDialog(True)
        if filepath == None: return
        data = FileLoader.loadGraphData(filepath)
        self.addGraphToScene(data)

    def importGraphFile(self):
        filePath , _= self.openFileDialog(True)
        if filePath == None: return
        data = FileLoader.loadGraphData(filePath)
        self.addGraphToScene(data)
    
    def importModule(self):
        # 実際は拡張子でフィルターしているだけ
        filePath,_ = self.openFileDialog(isLoad = True ,isModule = True) 
        if filePath == None: return
        data = FileLoader.loadGraphData(filePath)
        self.addModuleToScene(data ,filePath)

    ### SCENE ####
    def addGraphToScene(self,graphData:GraphData):
        nodeScene = self.nodeScene
        nodeGraph = nodeScene.nodeGraph
        ngc = nodeGraph.getController()
        graphModel = Serializer.dataToGraph(graphData)
        addGraphAction = CgpAction.ADD_GRAPH(graphModel)
        ngc.onCgpAction(addGraphAction)

    def addModuleToScene(self,graph,filePath):
        nodeScene = self.nodeScene
        nodeGraph = nodeScene.nodeGraph
        nodeGraph.addModuleFromGraphData(graph , filePath)

    def clean(self):
        # 警告を出す
        alert = QMessageBox()
        alert.setWindowTitle("Warning")
        alert.setText("シーンを消去します")
        alert.setInformativeText("よろしいですか？")
        # alert.setIcon(QMessageBox.Warning)
        alert.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        alert.setDefaultButton(QMessageBox.Cancel)
        CSound.play("error.wav")
        ret = alert.exec_()

        if ret == QMessageBox.Cancel:
            return False

        nodeGraph = self.nodeScene.nodeGraph
        nodeGraph.clearGraph()
        return True

    def onError(self, type,error):
        errortxt = error
        if(type == "AuthenticationError"):
            errortxt = "API key is invalid"
        # openAI rate limitter
        if type == openai.error.APIError:
            errortxt = "opne ai api rate limitter"

        alert = QMessageBox() 
        alert.setWindowTitle("Error")
        alert.setText(type)
        alert.setInformativeText(errortxt)
        # alert.setIcon(QMessageBox.Critical)
        CSound.play("error.wav")
        alert.exec_()

    def reload(self):
        isCanncelled = not self.clean()
        if isCanncelled: return
        filepath = self.lastOpenFile 
        data = FileLoader.loadGraphData(filepath)
        self.addGraphToScene(data)
    
    # 本当はwindow controllerみたいなのを作った方がいい
    def settingApiKey(self):
        settingView = self.editorPanel.settingView.show() 
        # get settingview tab index
        index = self.editorPanel.indexOf(settingView)
        self.editorPanel.setCurrentIndex(1)

    def setNameToCurrentTab(self, name):
        index = self.editorPanel.currentIndex()
        self.editorPanel.setTabText(index, name)

    def openFileDialog(self, isLoad = True , isModule = False):
        options = QFileDialog.Options()
        ext = ".json"
        # load 
        if isLoad: options |= QFileDialog.ReadOnly
        defaultNodePath =  Util.nodePath

        file_dialog_params = {
            True: {"name": "Load Module", "extMenu": "module (*.cgp);;All Files (*)"},
            False: {"name": "Load Graph", "extMenu": "graph (*.json);;All Files (*)"}
        }
        sp= file_dialog_params[isModule]
        qDialog = QFileDialog.getOpenFileName if isLoad else QFileDialog.getSaveFileName

        pathToOpen = self.lastOpenFile
        if isModule: pathToOpen = pathToOpen.replace(".json" , ".cgp")
        if pathToOpen is None:
            pathToOpen = defaultNodePath

        filePath, data = qDialog(self, sp["name"], pathToOpen, sp["extMenu"],options=options)
        # wait for dialog

        isCancel = filePath in {None ,""}
        if isCancel: return None , None
        else: 
            if not isModule: self.lastOpenFile = filePath
            filename = os.path.basename(filePath).replace(ext, "")
            self.setNameToCurrentTab(filename)
            return filePath , data

    def goAPIKeySetting(self):
        self.editorPanel.settingView.show() 
        # get settingview tab index
        self.editorPanel.setCurrentIndex(1)

    def goEditor(self):
        self.editorPanel.setCurrentIndex(0)

    def sendToChat(self , chatScreenName, answer , icon):
        self.rightPanel.chatView.send_message(chatScreenName, answer , icon)

    def getInspectorView(self):
        return self.leftPanel.inspectorView
    def getGraphView(self):
        return self.editorPanel.graphView
