VERSION 5.00
Object = "{831FDD16-0C5C-11D2-A9FC-0000F8754DA1}#2.2#0"; "MSCOMCTL.OCX"
Begin VB.Form Form1 
   Caption         =   "VFJ¶ÁÐ´Æ÷×ÛºÏ¹¤¾ß"
   ClientHeight    =   9210
   ClientLeft      =   60
   ClientTop       =   405
   ClientWidth     =   12375
   BeginProperty Font 
      Name            =   "Î¢ÈíÑÅºÚ"
      Size            =   9
      Charset         =   134
      Weight          =   400
      Underline       =   0   'False
      Italic          =   0   'False
      Strikethrough   =   0   'False
   EndProperty
   LinkTopic       =   "Form1"
   ScaleHeight     =   9210
   ScaleWidth      =   12375
   StartUpPosition =   3  '´°¿ÚÈ±Ê¡
   Begin VB.Frame TabStrip1__Tab1 
      BorderStyle     =   0  'None
      Caption         =   "»ù±¾¹¦ÄÜ"
      BeginProperty Font 
         Name            =   "ËÎÌå"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   2775
      Left            =   480
      TabIndex        =   34
      Top             =   2160
      Width           =   11415
      Begin VB.Frame Frame6 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2295
         Left            =   3960
         TabIndex        =   38
         Top             =   240
         Width           =   7095
         Begin VB.TextBox P1_Text_Diode3 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   5640
            TabIndex        =   75
            Text            =   "2"
            Top             =   600
            Width           =   255
         End
         Begin VB.TextBox P1_Text_Diode2 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   4560
            TabIndex        =   74
            Text            =   "2"
            Top             =   600
            Width           =   255
         End
         Begin VB.TextBox P1_Text_Diode1 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   3480
            TabIndex        =   73
            Text            =   "2"
            Top             =   600
            Width           =   255
         End
         Begin VB.TextBox P1_Text_Buzzer2 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   5640
            TabIndex        =   42
            Text            =   "2"
            Top             =   1500
            Width           =   255
         End
         Begin VB.TextBox P1_Text_Buzzer1 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   4080
            TabIndex        =   41
            Text            =   "2"
            Top             =   1500
            Width           =   255
         End
         Begin VB.CommandButton P1_Command_Diode 
            Caption         =   "·¢¹â¶þ¼«¹Ü"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   1080
            TabIndex        =   40
            Top             =   600
            Width           =   1335
         End
         Begin VB.CommandButton P1_Command_Buzzer 
            Caption         =   "·äÃùÆ÷"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   1080
            TabIndex        =   39
            Top             =   1440
            Width           =   1335
         End
         Begin VB.Label Label8 
            Caption         =   "Y"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   5280
            TabIndex        =   78
            Top             =   660
            Width           =   375
         End
         Begin VB.Label Label7 
            Caption         =   "G"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   4200
            TabIndex        =   77
            Top             =   660
            Width           =   375
         End
         Begin VB.Label Label6 
            Caption         =   "R"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   3120
            TabIndex        =   76
            Top             =   660
            Width           =   375
         End
         Begin VB.Label Label10 
            Caption         =   "Éùµ÷"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   5040
            TabIndex        =   44
            Top             =   1500
            Width           =   495
         End
         Begin VB.Label Label9 
            Caption         =   "·¢Òô´ÎÊý"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   3120
            TabIndex        =   43
            Top             =   1500
            Width           =   855
         End
      End
      Begin VB.Frame Frame5 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2295
         Left            =   480
         TabIndex        =   35
         Top             =   240
         Width           =   3135
         Begin VB.CommandButton P1_Command_ReaderVer 
            Caption         =   "¶ÁÐ´Æ÷°æ±¾²éÑ¯"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   615
            Left            =   600
            TabIndex        =   37
            Top             =   360
            Width           =   1815
         End
         Begin VB.CommandButton P1_Command_DllVersion 
            Caption         =   "¶¯Ì¬¿â°æ±¾²éÑ¯"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   615
            Left            =   600
            TabIndex        =   36
            Top             =   1320
            Width           =   1815
         End
      End
   End
   Begin VB.Frame TabStrip1__Tab2 
      BorderStyle     =   0  'None
      Caption         =   "¿¨Æ¬²âÊÔ"
      BeginProperty Font 
         Name            =   "Î¢ÈíÑÅºÚ"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   2775
      Left            =   480
      TabIndex        =   20
      Top             =   2160
      Visible         =   0   'False
      Width           =   11415
      Begin VB.Frame Frame7 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2535
         Left            =   4080
         TabIndex        =   25
         Top             =   120
         Width           =   7095
         Begin VB.CommandButton P2_Command_3F00 
            Caption         =   "Ñ¡Ôñ 3F00"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   240
            TabIndex        =   33
            Top             =   360
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_1001 
            Caption         =   "Ñ¡Ôñ 1001"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   1920
            TabIndex        =   32
            Top             =   360
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_1005 
            Caption         =   "¶Á 0015"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   1920
            TabIndex        =   31
            Top             =   1080
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_1006 
            Caption         =   "¶Á 0016"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   1920
            TabIndex        =   30
            Top             =   1800
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_DF01 
            Caption         =   "Ñ¡Ôñ DF01"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   3720
            TabIndex        =   29
            Top             =   360
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_EF01 
            Caption         =   "¶Á EF01"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   5400
            TabIndex        =   28
            Top             =   360
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_EF02 
            Caption         =   "¶Á EF02"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   5400
            TabIndex        =   27
            Top             =   1080
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_EF04 
            Caption         =   "¶Á EF04"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   495
            Left            =   5400
            TabIndex        =   26
            Top             =   1800
            Width           =   1335
         End
      End
      Begin VB.Frame Frame2 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2535
         Left            =   240
         TabIndex        =   21
         Top             =   120
         Width           =   3255
         Begin VB.CommandButton P2_Command_OpenCard 
            Caption         =   "´ò¿ª¿¨Æ¬"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   615
            Left            =   240
            TabIndex        =   24
            Top             =   480
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_CloseCard 
            Caption         =   "¹Ø±Õ¿¨Æ¬"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   615
            Left            =   240
            TabIndex        =   23
            Top             =   1440
            Width           =   1335
         End
         Begin VB.CommandButton P2_Command_RFreset 
            Caption         =   "ÉäÆµ¸´Î»"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   1575
            Left            =   1920
            TabIndex        =   22
            Top             =   480
            Width           =   975
         End
      End
   End
   Begin VB.Frame Frame4 
      Caption         =   "ÌìÏßÉèÖÃ"
      BeginProperty Font 
         Name            =   "ËÎÌå"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   1665
      Left            =   7560
      TabIndex        =   7
      Top             =   30
      Width           =   4575
      Begin VB.CheckBox P0_Check_antenna6 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   4080
         TabIndex        =   16
         Top             =   330
         Width           =   375
      End
      Begin VB.CheckBox P0_Check_antenna5 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   3648
         TabIndex        =   15
         Top             =   330
         Width           =   375
      End
      Begin VB.CheckBox P0_Check_antenna4 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   3216
         TabIndex        =   14
         Top             =   330
         Width           =   375
      End
      Begin VB.CheckBox P0_Check_antenna3 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   2784
         TabIndex        =   13
         Top             =   330
         Width           =   375
      End
      Begin VB.CheckBox P0_Check_antenna2 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   2352
         TabIndex        =   12
         Top             =   330
         Width           =   375
      End
      Begin VB.CheckBox P0_Check_antenna1 
         BeginProperty Font 
            Name            =   "ËÎÌå"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   1920
         TabIndex        =   11
         Top             =   330
         Width           =   375
      End
      Begin VB.ComboBox P0_Combo_antenna 
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   375
         ItemData        =   "¶ÁÐ´Æ÷¹¤¾ß.frx":0000
         Left            =   1920
         List            =   "¶ÁÐ´Æ÷¹¤¾ß.frx":0016
         TabIndex        =   10
         Text            =   "1 "
         Top             =   960
         Width           =   615
      End
      Begin VB.Label Label5 
         Caption         =   "Ñ¡Ôñ¶ÁÐ´Í·£º"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   375
         Left            =   480
         TabIndex        =   9
         Top             =   1080
         Width           =   1215
      End
      Begin VB.Label Label4 
         Caption         =   "ÌìÏßÂÖÑ¯ÉèÖÃ"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   375
         Left            =   480
         TabIndex        =   8
         Top             =   420
         Width           =   1215
      End
   End
   Begin VB.Frame Frame1 
      Caption         =   "»ù±¾ÉèÖÃ"
      BeginProperty Font 
         Name            =   "Î¢ÈíÑÅºÚ"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   1695
      Left            =   240
      TabIndex        =   1
      Top             =   0
      Width           =   7095
      Begin VB.CommandButton P0_Command_Find 
         Caption         =   "ËÑË÷¶ÁÐ´Æ÷"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   615
         Left            =   285
         TabIndex        =   72
         Top             =   960
         Width           =   3615
      End
      Begin VB.CommandButton P0_Command1 
         Caption         =   "´ò¿ª¶ÁÐ´Æ÷"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   615
         Left            =   4800
         TabIndex        =   18
         Top             =   240
         Width           =   1455
      End
      Begin VB.CommandButton P0_Command2 
         Caption         =   "¹Ø±Õ¶ÁÐ´Æ÷"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   615
         Left            =   4800
         TabIndex        =   4
         Top             =   960
         Width           =   1455
      End
      Begin VB.ComboBox P0_Combo_port 
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   375
         ItemData        =   "¶ÁÐ´Æ÷¹¤¾ß.frx":0032
         Left            =   2880
         List            =   "¶ÁÐ´Æ÷¹¤¾ß.frx":0051
         TabIndex        =   3
         Text            =   "COM1"
         Top             =   360
         Width           =   1020
      End
      Begin VB.ComboBox P0_Combo_region 
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   375
         ItemData        =   "¶ÁÐ´Æ÷¹¤¾ß.frx":008B
         Left            =   1080
         List            =   "¶ÁÐ´Æ÷¹¤¾ß.frx":0095
         TabIndex        =   2
         Text            =   "CQ"
         Top             =   360
         Width           =   780
      End
      Begin VB.Label Label2 
         Caption         =   "´®¿Ú£º"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   2280
         TabIndex        =   6
         Top             =   360
         Width           =   540
      End
      Begin VB.Label Label3 
         Caption         =   "µØÓò£º"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   420
         Left            =   360
         TabIndex        =   5
         Top             =   360
         Width           =   540
      End
   End
   Begin VB.Frame Frame3 
      Caption         =   "·µ»Ø£º"
      BeginProperty Font 
         Name            =   "Î¢ÈíÑÅºÚ"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   4095
      Left            =   240
      TabIndex        =   0
      Top             =   5040
      Width           =   11895
      Begin VB.TextBox P0_Text_Msg 
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   4095
         Left            =   0
         TabIndex        =   17
         Top             =   0
         Width           =   11895
      End
   End
   Begin MSComctlLib.TabStrip TabStrip1 
      Height          =   3255
      Left            =   240
      TabIndex        =   19
      Top             =   1800
      Width           =   11895
      _ExtentX        =   20981
      _ExtentY        =   5741
      TabWidthStyle   =   2
      TabFixedWidth   =   5292
      _Version        =   393216
      BeginProperty Tabs {1EFB6598-857C-11D1-B16A-00C0F0283628} 
         NumTabs         =   3
         BeginProperty Tab1 {1EFB659A-857C-11D1-B16A-00C0F0283628} 
            Caption         =   "»ù±¾¹¦ÄÜ"
            Key             =   "key1"
            ImageVarType    =   2
         EndProperty
         BeginProperty Tab2 {1EFB659A-857C-11D1-B16A-00C0F0283628} 
            Caption         =   "¿¨Æ¬²âÊÔ"
            Key             =   "key2"
            ImageVarType    =   2
         EndProperty
         BeginProperty Tab3 {1EFB659A-857C-11D1-B16A-00C0F0283628} 
            Caption         =   "¸ß¼¶¹¦ÄÜ"
            Key             =   "key3"
            ImageVarType    =   2
         EndProperty
      EndProperty
      BeginProperty Font {0BE35203-8F91-11CE-9DE3-00AA004BB851} 
         Name            =   "Î¢ÈíÑÅºÚ"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
   End
   Begin VB.Frame TabStrip1__Tab3 
      BorderStyle     =   0  'None
      Caption         =   "¸ß¼¶²âÊÔ"
      BeginProperty Font 
         Name            =   "ËÎÌå"
         Size            =   9
         Charset         =   134
         Weight          =   400
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   2775
      Left            =   480
      TabIndex        =   45
      Top             =   2160
      Visible         =   0   'False
      Width           =   11415
      Begin VB.Frame Frame9 
         Caption         =   "PSAM Test"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2655
         Left            =   0
         TabIndex        =   63
         Top             =   120
         Width           =   3735
         Begin VB.TextBox P3_Text_Command 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   270
            Left            =   720
            TabIndex        =   69
            Top             =   1560
            Width           =   420
         End
         Begin VB.TextBox Text7 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   735
            Left            =   0
            TabIndex        =   68
            Top             =   1920
            Width           =   3735
         End
         Begin VB.CommandButton P3_Command_RFpsam 
            Caption         =   "PSAM¸´Î»"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   67
            Top             =   420
            Width           =   1215
         End
         Begin VB.CommandButton P3_Command_PSAMcommand 
            Caption         =   "PSAMÖ¸Áî"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   1920
            TabIndex        =   66
            Top             =   960
            Width           =   1215
         End
         Begin VB.ComboBox P3_Combo_PSAMposition 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            ItemData        =   "¶ÁÐ´Æ÷¹¤¾ß.frx":00A1
            Left            =   2640
            List            =   "¶ÁÐ´Æ÷¹¤¾ß.frx":00B1
            TabIndex        =   65
            Text            =   "1 "
            Top             =   450
            Width           =   495
         End
         Begin VB.CommandButton P3_Command_CPUcommand 
            Caption         =   "CPUÖ¸Áî"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   64
            Top             =   960
            Width           =   1215
         End
         Begin VB.Label Label12 
            Caption         =   "Ö¸Áî£º"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   255
            Left            =   120
            TabIndex        =   71
            Top             =   1560
            Width           =   615
         End
         Begin VB.Label Label11 
            Caption         =   "¿¨²Û£º"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   255
            Left            =   1920
            TabIndex        =   70
            Top             =   473
            Width           =   615
         End
      End
      Begin VB.Frame Frame10 
         Caption         =   "¿¨Æ¬Á¬Ðø²âÊÔ"
         BeginProperty Font 
            Name            =   "Î¢ÈíÑÅºÚ"
            Size            =   9
            Charset         =   134
            Weight          =   400
            Underline       =   0   'False
            Italic          =   0   'False
            Strikethrough   =   0   'False
         EndProperty
         Height          =   2655
         Left            =   3840
         TabIndex        =   46
         Top             =   120
         Width           =   7575
         Begin VB.CommandButton P3_Command_StopTest 
            Caption         =   "Í£Ö¹²âÊÔ"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   735
            Left            =   6000
            TabIndex        =   55
            Top             =   1560
            Width           =   1335
         End
         Begin VB.CommandButton P3_Command_StartTest 
            Caption         =   "¿ªÊ¼²âÊÔ"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   735
            Left            =   6000
            TabIndex        =   54
            Top             =   480
            Width           =   1335
         End
         Begin VB.TextBox P3_Text_Blocks 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   1320
            TabIndex        =   53
            Text            =   "3"
            Top             =   2100
            Width           =   975
         End
         Begin VB.TextBox Text_StarBlock 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   1320
            TabIndex        =   52
            Text            =   "0"
            Top             =   1485
            Width           =   975
         End
         Begin VB.TextBox P3_Text_MAID 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   390
            Left            =   1320
            TabIndex        =   51
            Text            =   "3001"
            Top             =   870
            Width           =   975
         End
         Begin VB.TextBox P3_Text_Tested 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   390
            Left            =   4440
            TabIndex        =   50
            Text            =   "0"
            Top             =   2040
            Width           =   615
         End
         Begin VB.TextBox P3_Text_TestTotal 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   390
            Left            =   4440
            TabIndex        =   49
            Text            =   "100"
            Top             =   1200
            Width           =   615
         End
         Begin VB.TextBox P3_Text_TestTimes 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   390
            Left            =   4440
            TabIndex        =   48
            Text            =   "5"
            Top             =   360
            Width           =   615
         End
         Begin VB.TextBox P3_Text_ReadTimer 
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   390
            Left            =   1800
            TabIndex        =   47
            Text            =   "1"
            Top             =   360
            Width           =   495
         End
         Begin VB.Label Label16 
            Caption         =   "Blocks"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   62
            Top             =   2160
            Width           =   855
         End
         Begin VB.Label Label14 
            Caption         =   "StarBlock"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   61
            Top             =   1545
            Width           =   855
         End
         Begin VB.Label Label15 
            Caption         =   "MAID"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   60
            Top             =   930
            Width           =   615
         End
         Begin VB.Label Label19 
            Caption         =   "ÒÑ²âÊÔ¿¨Êý"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   255
            Left            =   3360
            TabIndex        =   59
            Top             =   2108
            Width           =   1095
         End
         Begin VB.Label Label18 
            Caption         =   "×Ü²âÊÔ¿¨Êý"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   255
            Left            =   3360
            TabIndex        =   58
            Top             =   1275
            Width           =   1095
         End
         Begin VB.Label Label17 
            Caption         =   "³¢ÊÔ´ÎÊý"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   3360
            TabIndex        =   57
            Top             =   435
            Width           =   1095
         End
         Begin VB.Label Label13 
            Caption         =   "ReadTimer(ms)"
            BeginProperty Font 
               Name            =   "Î¢ÈíÑÅºÚ"
               Size            =   9
               Charset         =   134
               Weight          =   400
               Underline       =   0   'False
               Italic          =   0   'False
               Strikethrough   =   0   'False
            EndProperty
            Height          =   375
            Left            =   240
            TabIndex        =   56
            Top             =   315
            Width           =   1335
         End
      End
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Private Sub TabStrip1_Click()
Select Case TabStrip1.SelectedItem.Key

Case "key1"
TabStrip1__Tab1.Visible = True
TabStrip1__Tab2.Visible = False
TabStrip1__Tab3.Visible = False

Case "key2"
TabStrip1__Tab1.Visible = False
TabStrip1__Tab2.Visible = True
TabStrip1__Tab3.Visible = False

Case "key3"
TabStrip1__Tab1.Visible = False
TabStrip1__Tab2.Visible = False
TabStrip1__Tab3.Visible = True
End Select
End Sub
