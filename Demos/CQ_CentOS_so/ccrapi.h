/************************************************************************/
/*																		*/
/*	CCRAPI.H															*/
/*                                                                      */
/*	System		:	CCR API DLL	 WIN32									*/
/* 	Program		: 	WIN 32 communication functions						*/
/*	Author		:	Paul Gor											*/
/*	Language	:	Borland C/C++ v5.02									*/
/*	Created		:	28-Jul-1997											*/
/*	Copyright	:	(c) HALCOM Systems Pty. Ltd.						*/
/*																		*/
/*	API Return code definition header file	 							*/
/*																		*/
/* 	Modification history												*/
/*																		*/
/*	Ver		Date		Who	Details	  									*/
/*  ------------------------------------------------------------------	*/
/*	1.0     28-Jul-1997 PG  Initial creation							*/
/*		   																*/
/************************************************************************/

/************************************************************************/
/*	 					RAPI Return Codes								*/
/************************************************************************/

#define		LRSUCCESS		0x00	//'Successful completion of request';
#define		LRSYSTEM		0x01	//'Unknown error';
#define		LRLASTCARD		0x02	//'Last Card Still Present';
#define		LRNOCARD		0x03	//'Card is not present';
#define		LRCTYPE			0x04	//'Card Type error';
#define		LRPARAM			0x05	//'Request Parameter error';
#define		LRACCESS		0x06	//'Card access error'
#define		LRREAD			0x07	//'Card read error';
#define		LRWRITE			0x08	//'Card write error';
#define		LRINCR			0x09	//'Purse increment error';
#define		LRDECR			0x0A	//'Purse decrement error';
#define		LRTRANSFER		0x0B	//'Purse value transfer error';
#define		LRRESTORE		0x0C	//'Purse restore error';
#define		LRPURSE			0x0D	//'Purse value corrupt';
#define		LRMADERR		0x0E	//'Card Directory error';
#define		LRFIXERR		0x0F	//'Purse fix error';
#define		LRFIXED			0x10	//'Purse found corrupt but fixed';
#define		LRNOTOPEN		0x11	//'Card not open';
#define		LRNOFILE		0x12	//'File not found';
#define		LRBADSIZE		0x13	//'Bad file size';
#define		LRABORTED		0x14	//'Request aborted';
#define		LRMANYCARD		0x15	//'Too many card present';
#define		LRFORMAT		0x16	//'Card format error';
#define		LRCREATE		0x17	//'Card file create error';
#define		LRDELETE		0x18	//'Card file delete error';
#define		LRALREADOPEN	0x19	//'Card has been opened already';
#define		LRALREADCLOSED	0x1A	//'Card has been closed already'
#define		LRMSTRKEYLOAD	0x1B	//'Cannot load master keys'
#define		LRAPPKEYLOAD	0x1C	//'Cannot load application keys'
#define		LRKEYCARD		0x1D	//'Keycard Error'
#define		LRUNFORMAT		0x1E	//'Card has files on it'
#define		LRFORMATTED		0x1F	//'Card has been formatted already'
#define		LRNOKBDCHAR		0x20	//'No keyboard character';
#define		LRNOTIMPL		0x7F	//'Function not implemented';
#define		LRUNKNOWN		0x80	//'Unknown error';

#define		LRMOREDATA		0xAA	//'More data to come';
#define		LRCCRBUSY		0xBB	//'Reader is busy';
#define		LRNOINIT		0xFF	//'Reader has not been opened';

#define		LRCRDNOTOPEN	0xFA	//'Card has not been opened';
#define		LRINUSE			0xFB	//'Card in use by another applications';
#define		LRAPPLICERR		0xFC	//'API system error';
#define		LRLINKLOST		0xFD	//'Link to Reader has been lost';
#define		LRBADCOMPORT	0xFE	//'COM port cannot be accessed';
#define		LRRESPLENGH		0xF9	//'Response length error';
#define		LRNOCRYPTBOX	0xF8	//'Crypto-Box not found';
#define		LRBADAPPACCESS	0xF7	//'Invalid Application access code';
#define		LRNOMAIDFILE	0xF6	//'Cannot open MAID definition file';
#define		LRBOXREAD		0xF5	//'Cannot read from Crypto-Box';
#define		LRBOXWRITE		0xF4	//'Cannot write to Crypto-Box';
#define		LRBOXNOKEYS		0xF3	//'No of Keys in Box is zero or invalid';
#define		LRSECURE		0xF2	//'Comms MAC checking failed';
#define		LRERRSELREADER	0xF1	//'Cannot change reader selection';

#define		LRNOSAM			0xC0	// No PSAM
#define		LRSAMACCESS		0xC1	// SAM access error
#define		LRMIFAREPRO		0xC5	// Mifare Pro is found
#define		LRM1S70CARD		0xC7

#define		RAPI_VERSION	207		// next version number of Reader API DLL
#define		CRAPI_VERSION	"V2.7"		// next version number of Reader API DLL

#define	ERR_PARAMS			-1000

