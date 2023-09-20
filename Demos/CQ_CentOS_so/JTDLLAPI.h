#ifndef JTDLLAPIH
#define JTDLLAPIH


#ifdef __cplusplus
extern "C" {
#endif

#if defined _WIN32 || defined __CYGWIN__
  #ifdef _USRDLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
  #define D_CALLTYPE __stdcall
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
  #define D_CALLTYPE
#endif


#ifndef	BYTE
typedef	unsigned char	BYTE ;
#endif
#ifndef	BOOL
typedef	unsigned char	BOOL;
#endif
#ifndef	FALSE
#define	FALSE			0
#endif
#ifndef	TRUE
#define	TRUE			!FALSE
#endif


DLL_PUBLIC int D_CALLTYPE	JT_OpenReader(int nMode, char *sParas1 ) ;
DLL_PUBLIC int D_CALLTYPE	JT_CloseReader(int nHandle ) ;
DLL_PUBLIC int D_CALLTYPE	JT_OpenCard( int nHandle, int *pCardPlace, char *sPhysicalCardNo );
DLL_PUBLIC int D_CALLTYPE	JT_CloseCard( int nHandle  );
DLL_PUBLIC int D_CALLTYPE	JT_LEDDisplay(int nHandle,unsigned char cRed,
										unsigned char cGreen, unsigned char cBlue) ;

DLL_PUBLIC int D_CALLTYPE	JT_AudioControl( int nHandle, unsigned char cTimes, unsigned char cVoice );

DLL_PUBLIC int D_CALLTYPE	JT_CPUCommand(int nHandle,char *pCommand,int iLenCmd,
										unsigned char *pReply,int *piLenRep) ;

DLL_PUBLIC int D_CALLTYPE	JT_SamCommand( int nHandle,int iSockID,char *pCommand,int iLenCmd,
										unsigned char *pReply,int *piLenRep) ;

DLL_PUBLIC int D_CALLTYPE	JT_SamReset( int nHandle, int iSockID, int iProtocolType ) ;

DLL_PUBLIC int D_CALLTYPE	JT_ResetRF( int nHandle );
DLL_PUBLIC int D_CALLTYPE	JT_ReadFile( int nHandle,int iFileId,int iKeyType,int iStartBlock,int iBlockNum,unsigned char *pReply) ;
DLL_PUBLIC int D_CALLTYPE	JT_WriteFile( int nHandle,int iFileId,int iKeyType,int iStartBlock,int iBlockNum,const unsigned char *pData);

DLL_PUBLIC int D_CALLTYPE	ReadCardFile( int nHandle, HANDLE AppInstance, PMAID AppFileId,
					BYTE StartBlock, BYTE BlockNo, LPBYTE BlocksRead,
					LPBYTE DataBlocks ) ;
DLL_PUBLIC int D_CALLTYPE	WriteCardFile( int nHandle, HANDLE AppInstance, PMAID AppFileId,
					BYTE StartBlock, LPBYTE DataBlocks, BYTE BlockNo,
					LPBYTE BlocksWritten );

DLL_PUBLIC int D_CALLTYPE	JT_GetStatus( int iHandle, int *pStatusCode );
DLL_PUBLIC int D_CALLTYPE	JT_GetStatusMsg( int nStatusCode, char *sStatusMsg, int nStatusMsgLen );
DLL_PUBLIC int D_CALLTYPE	JT_GetVersion( char *sVersion, int nVerLen ) ;
DLL_PUBLIC int D_CALLTYPE	JT_GetLastError( int nHandle ) ;
DLL_PUBLIC int D_CALLTYPE	JT_ReaderVersion( int nHandle, char * HardwareId, int iRHardwareId, 
										char * sReaderVersion, int iRVerMaxLength );
DLL_PUBLIC int D_CALLTYPE	JT_SetInitPPS( int nHandle, char *sPPS );
DLL_PUBLIC int D_CALLTYPE	JT_SetInitTimeOut( int nHandle, char *sTimeOut );

DLL_PUBLIC int D_CALLTYPE	JT_SelectRWUnit( int nHandle, BYTE Unit );
DLL_PUBLIC int D_CALLTYPE	JT_SetScanRWUnits( int nHandle, BYTE Units );

DLL_PUBLIC int D_CALLTYPE	read_cpc_mf(int handle, char* cpIssueId ,char* cpCardAppNo,char* version,char* startDate , char* expDate , char* power , char* rsswitch ) ;
DLL_PUBLIC int D_CALLTYPE	cpc_switchrs(int handle, char rsswitch );
DLL_PUBLIC int D_CALLTYPE	read_cpc_exfile(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	write_cpc_exfile(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	read_cpc_tracefile(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	read_cpc_feeinfo(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	read_cpc_ef03(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	write_cpc_ef03(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	card_extend_auth( int handle );
DLL_PUBLIC int D_CALLTYPE	set_cpc_clear(int handle );
DLL_PUBLIC int D_CALLTYPE	write_cpc_tracefile(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	write_cpc_feeinfo(int handle, int offset , char* data , int length );
DLL_PUBLIC int D_CALLTYPE	SelectMF( int handle );
DLL_PUBLIC int D_CALLTYPE	SelectDF( int handle );
DLL_PUBLIC int D_CALLTYPE	card_extend_auth2( int handle );
DLL_PUBLIC int D_CALLTYPE	psamStartUp( int handle );



#ifdef __cplusplus
}
#endif

#endif


