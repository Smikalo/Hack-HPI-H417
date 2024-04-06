import 'package:flutter/material.dart';
import 'dart:math' as math;

class FloatingLogoAnimation extends StatefulWidget {
  final Widget child;

  const FloatingLogoAnimation({super.key, required this.child});

  @override
  State<FloatingLogoAnimation> createState() => _FloatingLogoAnimationState();
}

class _FloatingLogoAnimationState extends State<FloatingLogoAnimation>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: false);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (BuildContext context, Widget? child) {
        return Transform.translate(
          offset: Offset(
            0.0,
            math.sin(_controller.value * math.pi * 2.0) * 10.0,
          ),
          child: child,
        );
      },
      child: widget.child,
    );
  }
}
