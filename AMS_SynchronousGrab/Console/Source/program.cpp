/*=============================================================================
  Copyright (C) 2013 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        program.cpp

  Description: Implementation of main entry point of SynchronousGrabConsole
               example of VimbaCPP.

-------------------------------------------------------------------------------

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#include <string>
#include <cstring>
#include <iostream>

#include "ApiController.h"
#include "Bitmap.h"

int main( int argc, char* argv[] )
{
	//std::cout << "starting main" << std::endl;
	std::string input; 
	std::cin >> input; 	// get picture name iteration from python PIPE
	
	
		
    VmbErrorType    err         = VmbErrorSuccess;

    char *          pCameraID   = NULL;             // The ID of the camera to use
    const char *    pFileName   = NULL;             // The filename for the bitmap to save
    bool            bPrintHelp  = false;            // Output help?
    int             i;                              // Counter for some iteration
    char *          pParameter;                     // The command line parameter

    //std::cout << "//////////////////////////////////////////\n";
    //std::cout << "///// Nate's Sweet Hi-jacked example /////\n";
    //std::cout << "//////////////////////////////////////////\n\n";


	if (input.empty()){
		pFileName = "NULL_file_name.bmp";
	} else { 
		//std::cout << "the input is: " << input << std::endl; 
		pFileName = input.c_str(); 
	}

    AVT::VmbAPI::Examples::ApiController apiController;
    
    //std::cout << "Vimba Version V" << apiController.GetVersion() << "\n";

    VmbFrameStatusType status = VmbFrameStatusIncomplete;
    err = apiController.StartUp(); //Startup the API
    if ( VmbErrorSuccess == err )
    {
        std::string strCameraID;
        AVT::VmbAPI::CameraPtrVector cameras = apiController.GetCameraList();
        if ( cameras.size() <= 0 ) // check if no cameras are detected
        {
            err = VmbErrorNotFound;
			std::cout << "Error: camera not found" << std::endl; 
        }
        else
        {
            err = cameras[0]->GetID( strCameraID ); // save the camera ID of the first camera in the cameraPtrVector in strCameraID
        }

        if ( VmbErrorSuccess == err ) // IMAGE CAPTURE (API SIDE) and IMAGE ACQUISITION (CAMERA SIDE) --------------------------
        {
            //std::cout << "Camera ID:" << strCameraID.c_str() << "\n\n";

            AVT::VmbAPI::FramePtr pFrame;
            err = apiController.AcquireSingleImage( strCameraID, pFrame );
            if ( VmbErrorSuccess == err )
            {
                err = pFrame->GetReceiveStatus( status );
                if (    VmbErrorSuccess == err
                     && VmbFrameStatusComplete == status )
                {
                    VmbPixelFormatType ePixelFormat = VmbPixelFormatMono8;
                    err = pFrame->GetPixelFormat( ePixelFormat );
                    if ( VmbErrorSuccess == err )
                    {
                        if (    ( VmbPixelFormatMono8 != ePixelFormat )
                            &&  ( VmbPixelFormatRgb8 != ePixelFormat ))
                        {
                            err = VmbErrorInvalidValue;
                        }
                        else
                        {
                            VmbUint32_t nImageSize = 0; 
                            err = pFrame->GetImageSize( nImageSize );
                            if ( VmbErrorSuccess == err )
                            {
                                VmbUint32_t nWidth = 0;
                                err = pFrame->GetWidth( nWidth );
                                if ( VmbErrorSuccess == err )
                                {
                                    VmbUint32_t nHeight = 0;
                                    err = pFrame->GetHeight( nHeight );
                                    if ( VmbErrorSuccess == err )
                                    {
                                        VmbUchar_t *pImage = NULL;
                                        err = pFrame->GetImage( pImage );
                                        if ( VmbErrorSuccess == err )
                                        {
                                            AVTBitmap bitmap;

                                            if ( VmbPixelFormatRgb8 == ePixelFormat )
                                            {
                                                bitmap.colorCode = ColorCodeRGB24;
                                            }
                                            else
                                            {
                                                bitmap.colorCode = ColorCodeMono8;
                                            }

                                            bitmap.bufferSize = nImageSize;
                                            bitmap.width = nWidth;
                                            bitmap.height = nHeight;

                                            // Create the bitmap
                                            if ( 0 == AVTCreateBitmap( &bitmap, pImage ))
                                            {
                                                std::cout << "Could not create bitmap.\n";
                                                err = VmbErrorResources;
                                            }
                                            else
                                            {
                                                // Save the bitmap
                                                if ( 0 == AVTWriteBitmapToFile( &bitmap, pFileName ))
                                                {
                                                    std::cout << "Could not write bitmap to file. SHOOT! \n";
                                                    err = VmbErrorOther;
                                                }
                                                else
                                                {
                                                    //std::cout << "Bitmap successfully written to file \"" << pFileName << "\"\n" ;
                                                    // Release the bitmap's buffer
                                                    if ( 0 == AVTReleaseBitmap( &bitmap ))
                                                    {
                                                        std::cout << "Could not release the bitmap.\n";
                                                        err = VmbErrorInternalFault;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        apiController.ShutDown();
    }

    if ( VmbErrorSuccess != err )
    {
        std::string strError = apiController.ErrorCodeToMessage( err );
        std::cout << "\nAn error occurred: " << strError.c_str() << "\n";
		std::cout << "error code: " << err << std::endl; 
    }
    if( VmbFrameStatusIncomplete == status)
    {
        std::cout<<"received frame was not complete\n";
    }
    

    return err;
}




