import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';

class AboutUsScreen extends StatelessWidget {
  const AboutUsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(20),
        alignment: Alignment.center,
        child: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 40),
              Text(
                'About us',
                textAlign: TextAlign.center,
                style: GoogleFonts.kadwa(
                  color: const Color(0xFFE6F0F3),
                  fontSize: 57,
                  height: 0.01,
                ),
              ),
              const SizedBox(height: 50),
              ConstrainedBox(
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height - 200,
                ),
                child: Text(
                  textAlign: TextAlign.center,
                  'Hello. We are the H417 team. This application was made to demonstrate the usecase of the "smooth route" technology we developed for the Hack HPI 2024 hackathon for the Starwit Technologies Challenge. \n',
                  style: GoogleFonts.kadwa(
                    color: Colors.white,
                    fontSize: 20,
                  ),
                ),
              ),
              Row(
                children: [
                  Image.asset("assets/image/Bitmap.png", scale: 3),
                  Container(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        Image.asset("assets/image/Bitmap (2).png", scale: 100),
                        const SizedBox(
                          height: 30,
                        ),
                        Text(
                          ' H417',
                          textAlign: TextAlign.center,
                          style: GoogleFonts.kadwa(
                            color: Colors.white,
                            fontSize: 28,
                          ),
                        ),
                        const SizedBox(height: 30),
                      ],
                    ),
                  ),
                ],
              ),
              Text(
                '\nThe essence of the technology is to apply an urban network of traffic cameras and artificial intelligence to generate a real-time map of traffic hazards in a city, predict accidents and better adapt cities \nfor self-driving cars. \n\n',
                style: GoogleFonts.kadwa(
                  color: Colors.white,
                ),
              ),
              Image.asset("assets/image/Bitmap (3).png", scale: 3),
              const SizedBox(
                height: 30,
              ),
              Text(
                'Using state-of-the-art research and current technologies in the fields of graph and convolutional neural networks, we efficiently process real-time traffic camera footage, compile an up-to-date road hazard map from the resulting data, and design a route that provides the driver with an optimal combination of speed and safety, simply put, smoothness.\n ',
                style: GoogleFonts.kadwa(
                  color: Colors.white,
                ),
              ),
              Image.asset("assets/image/Bitmap (4).png", scale: 3),
              const SizedBox(
                height: 30,
              ),
              Text.rich(
                TextSpan(
                  children: [
                    TextSpan(
                      text:
                          '\nThe results we obtain are available through a developer- and business-friendly API and can be applied to make the city safer and better adapt the urban environment for self-driving cars.\nYou can read more about technical details of the project on our ',
                      style: GoogleFonts.kadwa(
                        color: Colors.white,
                        fontSize: 15,
                      ),
                    ),
                    TextSpan(
                      recognizer: TapGestureRecognizer()
                        ..onTap = () async {
                          await launchUrl(Uri.parse(
                              'https://github.com/Smikalo/Hack-HPI-H417'));
                        },
                      text: 'GitHub',
                      style: GoogleFonts.kadwa(
                        color: const Color(0xFF456FFF),
                        fontSize: 15,
                        //textDecoration: TextDecoration.underline,
                      ),
                    ),
                    TextSpan(
                      text: '.',
                      style: GoogleFonts.kadwa(
                        color: Colors.white,
                        fontSize: 15,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
