import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class TransparentButton extends StatelessWidget {
  const TransparentButton({
    super.key,
    required this.onTap,
    required this.title,
  });

  final void Function() onTap;
  final String title;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
          side: const BorderSide(width: 2, color: Colors.white),
        ),
        elevation: 0,
        padding: EdgeInsets.zero,
        shadowColor: Colors.transparent,
        minimumSize: const Size(175, 57),
        alignment: Alignment.center,
      ),
      child: Text(
        title,
        textAlign: TextAlign.center,
        style: GoogleFonts.kadwa(
          color: Colors.white,
          fontSize: 24,
          height: 0.04,
        ),
      ),
    );
  }
}
