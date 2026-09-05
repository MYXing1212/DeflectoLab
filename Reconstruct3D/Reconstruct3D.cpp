// Reconstruct3D.cpp : 定义应用程序的类行为。
//

#include "stdafx.h"
#include "Reconstruct3D.h"
#include "MainDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CReconstruct3DApp, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()

CReconstruct3DApp::CReconstruct3DApp()
{

}

CReconstruct3DApp theApp;

BOOL CReconstruct3DApp::InitInstance()
{
	CWinApp::InitInstance();

	SetRegistryKey(_T("QUST"));

	CMainDlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
	}
	else if (nResponse == IDCANCEL)
	{
	}

	return FALSE;
}