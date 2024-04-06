import 'package:auth_login/animations/levitate_logo_animation.dart';
import 'package:auth_login/screens/login_screen.dart';
import 'package:auth_login/screens/register_screen.dart';
import 'package:auth_login/widgets/bright_button.dart';
import 'package:auth_login/widgets/transparent_button.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Hero(
            tag: "Image",
            child: FloatingLogoAnimation(
              child: Image.asset('assets/image/logo_transparent.png'),
            ),
          ),
          Text(
            'Deridefull',
            textAlign: TextAlign.center,
            style: GoogleFonts.kadwa(
              color: const Color(0xFFE6F0F3).withOpacity(0.2),
              fontSize: 48,
              height: 0.01,
            ),
          ),
          const SizedBox(height: 50),
          Text(
            'Welcome',
            textAlign: TextAlign.center,
            style: GoogleFonts.kadwa(
              color: const Color(0xFFE6F0F3),
              fontSize: 67,
            ),
          ),
          const SizedBox(height: 50),
          Text(
            'Please authorize',
            textAlign: TextAlign.center,
            style: GoogleFonts.kadwa(
              color: const Color(0xFFE6F0F3),
              fontSize: 32,
              height: 0.02,
            ),
          ),
          const SizedBox(height: 50),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              TransparentButton(
                onTap: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (ctx) => const RegisterScreen(),
                    ),
                  );
                },
                title: "Register",
              ),
              BrightButton(
                onTap: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (ctx) => const LoginScreen(),
                    ),
                  );
                },
                title: "Login",
              ),
            ],
          )
        ],
      ),
    );
  }
}
